/*
 * MLP encoder
 * Copyright (c) 2008 Ramiro Polla <ramiro@lisha.ufsc.br>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "avcodec.h"
#include "bitstream.h"
#include "libavutil/crc.h"

/* TODO add comments! */

#define MAJOR_HEADER_INTERVAL 16

#define MAX_CHANNELS        16
#define MAX_SUBSTREAMS      2
#define MAX_SAMPLERATE      192000
#define MAX_BLOCKSIZE       (40 * (MAX_SAMPLERATE / 48000))
#define MAX_BLOCKSIZE_POW2  (64 * (MAX_SAMPLERATE / 48000))

#define MAX_FILTER_ORDER    8
#define NUM_FILTERS         2

#define FIR 0
#define IIR 1

typedef struct {
    uint8_t         min_channel;
    uint8_t         max_channel;
    uint8_t         max_matrix_channel;

    uint8_t         noise_shift;
    uint32_t        noisegen_seed;

    int             data_check_present;

    int32_t         lossless_check_data;
} RestartHeader;

typedef struct {
    uint16_t        blocksize;
    uint8_t         quant_step_size[MAX_CHANNELS];

    uint8_t         num_primitive_matrices;

    int8_t          output_shift[MAX_CHANNELS];

    uint8_t         param_presence_flags;
#define PARAM_PRESENCE_FLAGS    (1 << 8)

#define PARAMS_DEFAULT      (0xFF)
#define PARAM_BLOCKSIZE     (1 << 7)
#define PARAM_MATRIX        (1 << 6)
#define PARAM_OUTSHIFT      (1 << 5)
#define PARAM_QUANTSTEP     (1 << 4)
#define PARAM_FIR           (1 << 3)
#define PARAM_IIR           (1 << 2)
#define PARAM_HUFFOFFSET    (1 << 1)

    uint8_t         codebook[MAX_CHANNELS];
    uint8_t         huff_lsbs[MAX_CHANNELS];

    /* TODO This should be part of the greater context. */
    int16_t         huff_offset[MAX_CHANNELS];
#define HUFF_OFFSET_MIN    -16384
#define HUFF_OFFSET_MAX     16383

} DecodingParams;

typedef struct {
    uint8_t         order;
    uint8_t         shift;
    int32_t         coeff[MAX_FILTER_ORDER];
    int32_t         state[MAX_FILTER_ORDER];
} FilterParams;

typedef struct {
    AVCodecContext *avctx;

    int             num_substreams;

    int             sample_fmt;
    int             sample_rate;

    int32_t         sample_buffer[MAX_BLOCKSIZE][MAX_CHANNELS+2];

    uint16_t        timestamp;

    uint8_t         mlp_channels;

    FilterParams    filter_params[MAX_CHANNELS][NUM_FILTERS];

    DecodingParams  decoding_params[MAX_SUBSTREAMS];
    RestartHeader   restart_header[MAX_SUBSTREAMS];
} MLPEncodeContext;

#define SYNC_MAJOR      0xf8726f

#define SYNC_MLP        0xbb
#define SYNC_TRUEHD     0xba

#define BITS_16         0x0
#define BITS_20         0x1
#define BITS_24         0x2

#define MAX_SAMPLERATE  192000

static int mlp_sample_rate(int sample_rate)
{
    int sample_base = 48000;
    uint8_t code = 0x0;

    switch (sample_rate) {
    case 44100 << 0:
    case 44100 << 1:
    case 44100 << 2:
        sample_base = 44100;
        code = 0x8;
    case 48000 << 0:
    case 48000 << 1:
    case 48000 << 2:
        break;
    default:
        return -1;
    }

    for (; sample_rate != sample_base; sample_rate >>= 1)
        code++;

    return code;
}

/* TODO all these checksum functions and crc stuff can be shared between
 * encoder and decoder. */

static AVCRC crc_1D[1024];
static AVCRC crc_2D[1024];
static AVCRC crc_63[1024];

static uint16_t mlp_checksum16(const uint8_t *buf, unsigned int buf_size)
{
    uint16_t crc = av_crc(crc_2D, 0, buf, buf_size - 2);

    crc ^= AV_RL16(buf + buf_size - 2);

    return crc;
}

static uint8_t mlp_checksum8(const uint8_t *buf, unsigned int buf_size)
{
    uint8_t checksum = av_crc(crc_63, 0x3c, buf, buf_size - 1); // crc_63[0xa2] == 0x3c
    checksum ^= buf[buf_size-1];
    return checksum;
}

static uint8_t mlp_restart_checksum(const uint8_t *buf, unsigned int bit_size)
{
    int i;
    int num_bytes = (bit_size + 2) / 8;

    int crc = crc_1D[buf[0] & 0x3f];
    crc = av_crc(crc_1D, crc, buf + 1, num_bytes - 2);
    crc ^= buf[num_bytes - 1];

    for (i = 0; i < ((bit_size + 2) & 7); i++) {
        crc <<= 1;
        if (crc & 0x100)
            crc ^= 0x11D;
        crc ^= (buf[num_bytes] >> (7 - i)) & 1;
    }

    return crc;
}

static uint8_t calculate_parity(const uint8_t *buf, unsigned int buf_size)
{
    uint32_t scratch = 0;
    const uint8_t *buf_end = buf + buf_size;

    for (; buf < buf_end - 3; buf += 4)
        scratch ^= *((const uint32_t*)buf);

    scratch ^= scratch >> 16;
    scratch ^= scratch >> 8;

    for (; buf < buf_end; buf++)
        scratch ^= *buf;

    return scratch;
}

static void write_major_sync(MLPEncodeContext *ctx, uint8_t *buf, int buf_size)
{
    PutBitContext pb;

    init_put_bits(&pb, buf, buf_size);

    put_bits(&pb, 24, SYNC_MAJOR       );
    put_bits(&pb,  8, SYNC_MLP         );
    put_bits(&pb,  4, ctx->sample_fmt  );
    put_bits(&pb,  4, ctx->sample_fmt  );
    put_bits(&pb,  4, ctx->sample_rate );
    put_bits(&pb,  4, ctx->sample_rate );
    put_bits(&pb, 11, 0                );
    put_bits(&pb,  5, ctx->mlp_channels);

    /* TODO copied from luckynight.mlp, 440hz.mlp and god.mlp. */
    put_bits(&pb, 16, 0xb752           );
    put_bits(&pb, 16, 0x4000           );
    put_bits(&pb, 16, 0                );

    put_bits(&pb,  1, 1                ); /* TODO is_vbr */
    put_bits(&pb, 15, 0                ); /* TODO peak_bitrate */
    put_bits(&pb,  4, 1                ); /* TODO num_substreams */

    /* TODO copied from luckynight.mlp, 440hz.mlp. */
#if 0
god   20d763f0000808000004536
440hz
lucky 1054c0300008080001b538c
#endif
    put_bits(&pb,  4, 0x1              );
    put_bits(&pb, 32, 0x054c0300       );
    put_bits(&pb, 32, 0x00808000       );
    put_bits(&pb,  8, 0x1b             );

    flush_put_bits(&pb);

    AV_WL16(buf+26, mlp_checksum16(buf, 26));
}

static void write_restart_header(MLPEncodeContext *ctx,
                                 PutBitContext *pb, int substr)
{
    RestartHeader *rh = &ctx->restart_header[substr];
    int32_t lossless_check = rh->lossless_check_data;
    unsigned int start_count = put_bits_count(pb);
    PutBitContext tmpb;
    uint8_t checksum;
    unsigned int ch;

    lossless_check ^= lossless_check >> 16;
    lossless_check ^= lossless_check >>  8;
    lossless_check &= 0xFF;

    put_bits(pb, 14, 0x31ea                ); /* TODO 0x31eb */
    put_bits(pb, 16, 0                     ); /* TODO I don't know what this is. Ask Ian. */
    put_bits(pb,  4, rh->min_channel       );
    put_bits(pb,  4, rh->max_channel       );
    put_bits(pb,  4, rh->max_matrix_channel);
    put_bits(pb,  4, rh->noise_shift       );
    put_bits(pb, 23, rh->noisegen_seed     );
    put_bits(pb, 19, 0                     ); /* TODO What the hell is this? */
    put_bits(pb,  1, rh->data_check_present);
    put_bits(pb,  8, lossless_check        );
    put_bits(pb, 16, 0                     ); /* this is zero =) */

    for (ch = 0; ch <= rh->max_matrix_channel; ch++)
        put_bits(pb, 6, ch);

    /* data must be flushed for the checksum to be right. */
    tmpb = *pb;
    flush_put_bits(&tmpb);

    checksum = mlp_restart_checksum(pb->buf, put_bits_count(pb) - start_count);

    put_bits(pb,  8, checksum);
}

static av_cold int mlp_encode_init(AVCodecContext *avctx)
{
    MLPEncodeContext *ctx = avctx->priv_data;
    unsigned int substr;

    ctx->avctx = avctx;

    ctx->sample_rate = mlp_sample_rate(avctx->sample_rate);
    if (ctx->sample_rate < 0) {
        av_log(avctx, AV_LOG_ERROR, "Unsupported sample_rate.\n");
        return -1;
    }

    /* TODO support more channels. */
    if (avctx->channels > 2) {
        av_log(avctx, AV_LOG_ERROR,
               "Only mono and stereo are supported at the moment.\n");
        return -1;
    }

    switch (avctx->sample_fmt) {
    case SAMPLE_FMT_S16:    ctx->sample_fmt = BITS_16; break;
    /* TODO 20 bits: */
    /* TODO Find out how to actually support 24 bits and update all occurences
     * of hardcoded 8s with appropriate value (probably quant_step_size). */
    case SAMPLE_FMT_S24:    ctx->sample_fmt = BITS_24; break;
    default:
        av_log(avctx, AV_LOG_ERROR, "Sample format not supported.\n");
        return -1;
    }

    avctx->frame_size               = 40 << (ctx->sample_rate & 0x7);
    avctx->coded_frame              = avcodec_alloc_frame();
    avctx->coded_frame->key_frame   = 1;

    av_crc_init(crc_1D, 0,  8,   0x1D, sizeof(crc_1D));
    av_crc_init(crc_2D, 0, 16, 0x002D, sizeof(crc_2D));
    av_crc_init(crc_63, 0,  8,   0x63, sizeof(crc_63));

    /* TODO mlp_channels is more complex, but for now
     * we only accept mono and stereo. */
    ctx->mlp_channels   = avctx->channels - 1;
    ctx->num_substreams = 1;

    for (substr = 0; substr < ctx->num_substreams; substr++) {
        DecodingParams *dp = &ctx->decoding_params[substr];
        RestartHeader  *rh = &ctx->restart_header [substr];
        uint8_t param_presence_flags = 0;
        unsigned int channel;

        rh->min_channel        = 0;
        rh->max_channel        = avctx->channels - 1;
        rh->max_matrix_channel = 1;

        rh->noise_shift        = 0;
        rh->noisegen_seed      = 0;

        rh->data_check_present = 0;

        dp->blocksize          = avctx->frame_size;

        for (channel = 0; channel <= rh->max_channel; channel++) {
            dp->quant_step_size[channel] = 8;
            dp->codebook       [channel] = 0;
            dp->huff_lsbs      [channel] = 24;
        }

        param_presence_flags |= PARAM_BLOCKSIZE;
/*      param_presence_flags |= PARAM_MATRIX; */
/*      param_presence_flags |= PARAM_OUTSHIFT; */
        param_presence_flags |= PARAM_QUANTSTEP;
        param_presence_flags |= PARAM_FIR;
        param_presence_flags |= PARAM_IIR;
        param_presence_flags |= PARAM_HUFFOFFSET;

        dp->param_presence_flags = param_presence_flags;
    }

    return 0;
}

static int inline number_sbits(int number)
{
    int bits = 0;

    if      (number > 0)
        for (bits = 31; bits && !(number & (1<<(bits-1))); bits--);
    else if (number < 0)
        for (bits = 31; bits &&  (number & (1<<(bits-1))); bits--);

    return bits + 1;
}

static void code_filter_coeffs(MLPEncodeContext *ctx,
                               unsigned int channel, unsigned int filter,
                               int *pcoeff_shift, int *pcoeff_bits)
{
    FilterParams *fp = &ctx->filter_params[channel][filter];
    int min = INT_MAX, max = INT_MIN;
    int bits, shift;
    int or = 0;
    int order;

    for (order = 0; order < fp->order; order++) {
        int coeff = fp->coeff[order];

        if (coeff < min)
            min = coeff;
        if (fp->coeff[order] > max)
            max = coeff;

        or |= coeff;
    }

    bits = FFMAX(number_sbits(min), number_sbits(max));

    for (shift = 0; shift < 7 && !(or & (1<<shift)); shift++);

    *pcoeff_bits  = bits;
    *pcoeff_shift = shift;
}

static void write_filter_params(MLPEncodeContext *ctx, PutBitContext *pb,
                                unsigned int channel, unsigned int filter)
{
    FilterParams *fp = &ctx->filter_params[channel][filter];

    put_bits(pb, 4, fp->order);

    if (fp->order > 0) {
        int coeff_shift;
        int coeff_bits;
        int i;

        code_filter_coeffs(ctx, channel, filter, &coeff_shift, &coeff_bits);

        put_bits(pb, 4, fp->shift  );
        put_bits(pb, 5, coeff_bits );
        put_bits(pb, 3, coeff_shift);

        for (i = 0; i < fp->order; i++) {
            int coeff = fp->coeff[i] >> coeff_shift;

            put_sbits(pb, coeff_bits, coeff);
        }

        put_bits(pb, 1, 0);
    }
}

static void write_decoding_params(MLPEncodeContext *ctx, PutBitContext *pb,
                                  unsigned int substr, int params_changed)
{
    DecodingParams *dp = &ctx->decoding_params[substr];
    RestartHeader  *rh = &ctx->restart_header [substr];
    unsigned int ch;

    if (dp->param_presence_flags != PARAMS_DEFAULT &&
        params_changed & PARAM_PRESENCE_FLAGS) {
        put_bits(pb, 1, 1);
        put_bits(pb, 8, dp->param_presence_flags);
    } else {
        put_bits(pb, 1, 0);
    }

    if (dp->param_presence_flags & PARAM_BLOCKSIZE) {
        if (params_changed       & PARAM_BLOCKSIZE) {
            put_bits(pb, 1, 1);
            put_bits(pb, 9, dp->blocksize);
        } else {
            put_bits(pb, 1, 0);
        }
    }

    if (dp->param_presence_flags & PARAM_MATRIX) {
        if (params_changed       & PARAM_MATRIX) {
            put_bits(pb, 1, 1);
#if 1
            put_bits(pb, 4, 0);
#else
            /* TODO no primitive matrices yet. */
            put_bits(pb, 4, dp->num_primitive_matrices);
#endif
        } else {
            put_bits(pb, 1, 0);
        }
    }

    if (dp->param_presence_flags & PARAM_OUTSHIFT) {
        if (params_changed       & PARAM_OUTSHIFT) {
            put_bits(pb, 1, 1);
            for (ch = 0; ch <= rh->max_matrix_channel; ch++)
                put_sbits(pb, 4, dp->output_shift[ch]);
        } else {
            put_bits(pb, 1, 0);
        }
    }

    if (dp->param_presence_flags & PARAM_QUANTSTEP) {
        if (params_changed       & PARAM_QUANTSTEP) {
            put_bits(pb, 1, 1);
            for (ch = 0; ch <= rh->max_channel; ch++)
                put_bits(pb, 4, dp->quant_step_size[ch]);
        } else {
            put_bits(pb, 1, 0);
        }
    }

    for (ch = rh->min_channel; ch <= rh->max_channel; ch++) {
        if (dp->param_presence_flags & 0xF) {
            put_bits(pb, 1, 1);

            if (dp->param_presence_flags & PARAM_FIR) {
                if (params_changed       & PARAM_FIR) {
                    put_bits(pb, 1, 1);
                    write_filter_params(ctx, pb, ch, FIR);
                } else {
                    put_bits(pb, 1, 0);
                }
            }

            if (dp->param_presence_flags & PARAM_IIR) {
                if (params_changed       & PARAM_IIR) {
                    put_bits(pb, 1, 1);
                    write_filter_params(ctx, pb, ch, IIR);
                } else {
                    put_bits(pb, 1, 0);
                }
            }

            if (dp->param_presence_flags & PARAM_HUFFOFFSET) {
                if (params_changed       & PARAM_HUFFOFFSET) {
                    put_bits(pb,  1, 1);
                    put_sbits(pb, 15, dp->huff_offset[ch]);
                } else {
                    put_bits(pb, 1, 0);
                }
            }

            put_bits(pb, 2, dp->codebook [ch]);
            put_bits(pb, 5, dp->huff_lsbs[ch]);
        } else {
            put_bits(pb, 1, 0);
        }
    }
}

static void input_data(MLPEncodeContext *ctx, const short *samples,
                       int32_t *lossless_check_data)
{
    unsigned int substr;

    for (substr = 0; substr < ctx->num_substreams; substr++) {
        DecodingParams *dp = &ctx->decoding_params[substr];
        RestartHeader  *rh = &ctx->restart_header [substr];
        int32_t lossless_check_data_temp = 0;
        unsigned int channel;
        int i;

        for (channel = 0; channel <= rh->max_channel; channel++) {
            for (i = 0; i < dp->blocksize; i++) {
                int32_t sample = samples[i * (rh->max_channel + 1) + channel];
                sample <<= 8;
                lossless_check_data_temp ^= (sample & 0x00ffffff) << channel;
                ctx->sample_buffer[i][channel] = sample;
            }
        }

        lossless_check_data[substr] = lossless_check_data_temp;
    }
}

static void set_filter_params(MLPEncodeContext *ctx,
                              unsigned int channel, unsigned int filter)
{
    FilterParams *fp = &ctx->filter_params[channel][filter];

    if (filter == FIR) {
        fp->order    =  4;
        fp->shift    =  0;
        fp->coeff[0] =  1;
        fp->coeff[1] =  0;
        fp->coeff[2] =  0;
        fp->coeff[3] =  0;
    } else { /* IIR */
        fp->order    =  4;
        fp->shift    =  0;
        fp->coeff[0] =  0;
        fp->coeff[1] =  0;
        fp->coeff[2] =  0;
        fp->coeff[3] =  0;
    }
}

#define MSB_MASK(bits)  (-1u << bits)

static void apply_filter(MLPEncodeContext *ctx, unsigned int channel)
{
    int32_t filter_state_buffer[NUM_FILTERS][MAX_BLOCKSIZE + MAX_FILTER_ORDER];
    FilterParams *fp[NUM_FILTERS] = { &ctx->filter_params[channel][FIR],
                                      &ctx->filter_params[channel][IIR], };
    int32_t mask = MSB_MASK(8); /* TODO quant_step_size */
    unsigned int filter_shift = fp[FIR]->shift;
    int index = MAX_BLOCKSIZE;
    int filter;
    int i;

    for (filter = 0; filter < NUM_FILTERS; filter++) {
        memcpy(&filter_state_buffer[filter][MAX_BLOCKSIZE],
               &fp[filter]->state[0],
               MAX_FILTER_ORDER * sizeof(int32_t));
    }

    for (i = 0; i < ctx->avctx->frame_size; i++) {
        int32_t sample = ctx->sample_buffer[i][channel];
        unsigned int order;
        int64_t accum = 0;
        int32_t residual;

        for (filter = 0; filter < NUM_FILTERS; filter++)
            for (order = 0; order < fp[filter]->order; order++)
                accum += (int64_t)filter_state_buffer[filter][index + order] *
                         fp[filter]->coeff[order];

        accum  >>= filter_shift;
        residual = sample - (accum & mask);

        --index;

        filter_state_buffer[FIR][index] = sample;
        filter_state_buffer[IIR][index] = residual;

        /* Store residual. */
        ctx->sample_buffer[i][channel] = residual;
    }

    for (filter = 0; filter < NUM_FILTERS; filter++) {
        memcpy(&fp[filter]->state[0],
               &filter_state_buffer[filter][index],
               MAX_FILTER_ORDER * sizeof(int32_t));
    }
}

static const uint8_t huffman_bits0[] = {
    9, 8, 7, 6, 5, 4, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9,
};

static const uint8_t huffman_bits1[] = {
    9, 8, 7, 6, 5, 4, 3,    2, 2,    3, 4, 5, 6, 7, 8, 9,
};

static const uint8_t huffman_bits2[] = {
    9, 8, 7, 6, 5, 4, 3,      1,     3, 4, 5, 6, 7, 8, 9,
};

static const uint8_t *huffman_bits[] = {
    huffman_bits0, huffman_bits1, huffman_bits2,
};

static const uint8_t huffman_bitcount2[] = {
    9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3,
    1, 1,
    3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9,
};

static const uint8_t *huffman_bitcount[] = {
    huffman_bits0, huffman_bits1, huffman_bitcount2,
};

static const uint8_t huffman_codes0[] = {
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x04, 0x05, 0x06, 0x07,
    0x03, 0x05, 0x09, 0x11, 0x21, 0x41, 0x81,
};

static const uint8_t huffman_codes1[] = {
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x02, 0x03,
    0x03, 0x05, 0x09, 0x11, 0x21, 0x41, 0x81,
};

static const uint8_t huffman_codes2[] = {
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01,
    0x03, 0x05, 0x09, 0x11, 0x21, 0x41, 0x81,
};

static const uint8_t *huffman_codes[] = {
    huffman_codes0, huffman_codes1, huffman_codes2,
};

static int codebook_extremes[3][2] = {
    {-9, 8}, {-8, 7}, {-15, 14},
};

static int codebook_offsets[3] = {
    9, 8, 7,
};

typedef struct BestOffset {
    int16_t offset;
    int bitcount;
    int lsb_bits;
} BestOffset;

static void no_codebook_bits(MLPEncodeContext *ctx, unsigned int substr,
                             unsigned int channel,
                             int16_t min, int16_t max,
                             BestOffset *bo)
{
    DecodingParams *dp = &ctx->decoding_params[substr];
    int16_t offset, unsign;
    uint16_t diff;
    int lsb_bits;

    /* Set offset inside huffoffset's boundaries by adjusting extremes
     * so that more bits are used thus shifting the offset. */
    if (min < HUFF_OFFSET_MIN)
        max = FFMAX(max, HUFF_OFFSET_MIN + HUFF_OFFSET_MIN - min + 1);
    if (max > HUFF_OFFSET_MAX)
        min = FFMIN(min, HUFF_OFFSET_MAX + HUFF_OFFSET_MAX - max - 1);

    /* Determine offset and minimum number of bits. */
    diff = max - min;

    lsb_bits = number_sbits(diff) - 1;

    unsign = 1 << (lsb_bits - 1);

    /* If all samples are the same (lsb_bits == 0), offset must be
     * adjusted because of sign_shift. */
    offset = min + diff / 2 + !!lsb_bits;

    /* Check if we can use the same offset as last access_unit to save
     * on writing a new header. */
    if (lsb_bits + 8 == dp->huff_lsbs[channel]) {
        int16_t cur_offset = dp->huff_offset[channel];
        int16_t cur_max    = cur_offset + unsign - 1;
        int16_t cur_min    = cur_offset - unsign;

        if (min > cur_min && max < cur_max)
            offset = cur_offset;
    }

    bo->offset   = offset;
    bo->lsb_bits = lsb_bits;
    bo->bitcount = lsb_bits * dp->blocksize;
}

static inline void codebook_bits_offset(MLPEncodeContext *ctx, unsigned int substr,
                                 unsigned int channel, int codebook,
                                 int32_t min, int32_t max, int16_t offset,
                                 BestOffset *bo, int *pnext, int up)
{
    DecodingParams *dp = &ctx->decoding_params[substr];
    int32_t codebook_min = codebook_extremes[codebook][0];
    int32_t codebook_max = codebook_extremes[codebook][1];
    int codebook_offset  = -codebook_min;
    int lsb_bits = 0, bitcount = 0;
    int next = INT_MAX;
    int unsign, mask;
    int i;

    min -= offset;
    max -= offset;

    while (min < codebook_min || max > codebook_max) {
        lsb_bits++;
        min >>= 1;
        max >>= 1;
    }

    unsign = 1 << lsb_bits;
    mask   = unsign - 1;

    for (i = 0; i < dp->blocksize; i++) {
        int32_t sample = ctx->sample_buffer[i][channel] >> 8;
        int temp_next;

        sample -= offset;

        if (up)
            temp_next = unsign - (sample & mask);
        else
            temp_next = (sample & mask) + 1;

        if (temp_next < next)
            next = temp_next;

        sample >>= lsb_bits;

        bitcount += huffman_bitcount[codebook][sample + codebook_offset];
    }

    if (codebook == 2)
        lsb_bits++;

    bo->offset   = offset;
    bo->lsb_bits = lsb_bits;
    bo->bitcount = lsb_bits * dp->blocksize + bitcount;

    *pnext       = next;
}

static inline void codebook_bits(MLPEncodeContext *ctx, unsigned int substr,
                          unsigned int channel, int codebook,
                          int average, int16_t min, int16_t max,
                          BestOffset *bo, int direction)
{
    int previous_count = INT_MAX;
    int offset_min, offset_max;
    int is_greater = 0;
    int offset = av_clip(average, HUFF_OFFSET_MIN, HUFF_OFFSET_MAX);
    int next;

    offset_min = FFMAX(min, HUFF_OFFSET_MIN);
    offset_max = FFMIN(max, HUFF_OFFSET_MAX);

    for (;;) {
        BestOffset temp_bo;

        codebook_bits_offset(ctx, substr, channel, codebook,
                                min, max, offset,
                                &temp_bo, &next, direction);

        if (temp_bo.bitcount < previous_count) {
            if (temp_bo.bitcount < bo->bitcount)
                *bo = temp_bo;

            is_greater = 0;
        } else if (++is_greater >= 3)
            break;

        previous_count = temp_bo.bitcount;

        if (direction) {
            offset += next;
            if (offset > offset_max)
                break;
        } else {
            offset -= next;
            if (offset < offset_min)
                break;
        }
    }
}

static void determine_bits(MLPEncodeContext *ctx, unsigned int substr)
{
    DecodingParams *dp = &ctx->decoding_params[substr];
    RestartHeader  *rh = &ctx->restart_header [substr];
    unsigned int channel;

    for (channel = 0; channel <= rh->max_channel; channel++) {
        int16_t min = INT16_MAX, max = INT16_MIN;
        int best_codebook = 0;
        BestOffset bo;
        int average = 0;
        int i;

        /* Determine extremes and average. */
        for (i = 0; i < dp->blocksize; i++) {
            int32_t sample = ctx->sample_buffer[i][channel] >> 8;
            if (sample < min)
                min = sample;
            if (sample > max)
                max = sample;
            average += sample;
        }
        average /= dp->blocksize;

        no_codebook_bits(ctx, substr, channel, min, max, &bo);

        for (i = 1; i < 4; i++) {
            BestOffset temp_bo = { 0, INT_MAX, 0, };

            codebook_bits(ctx, substr, channel, i - 1, average,
                            min, max, &temp_bo, 0);
            codebook_bits(ctx, substr, channel, i - 1, average,
                            min, max, &temp_bo, 1);

            if (temp_bo.bitcount < bo.bitcount) {
                bo = temp_bo;
                best_codebook = i;
            }
        }

        /* Update context. */
        dp->huff_offset[channel] = bo.offset;
        dp->huff_lsbs  [channel] = bo.lsb_bits + 8;
        dp->codebook   [channel] = best_codebook;
    }
}

static void write_block_data(MLPEncodeContext *ctx, PutBitContext *pb,
                             unsigned int substr)
{
    DecodingParams *dp = &ctx->decoding_params[substr];
    RestartHeader  *rh = &ctx->restart_header [substr];
    int codebook_offset[MAX_CHANNELS];
    int codebook[MAX_CHANNELS];
    int16_t unsign[MAX_CHANNELS];
    int16_t offset[MAX_CHANNELS];
    int lsb_bits[MAX_CHANNELS];
    unsigned int i, ch;

    for (ch = rh->min_channel; ch <= rh->max_channel; ch++) {
        lsb_bits       [ch] = dp->huff_lsbs  [ch] - dp->quant_step_size[ch];
        codebook       [ch] = dp->codebook   [ch] - 1;
        offset         [ch] = dp->huff_offset[ch];
        codebook_offset[ch] = codebook_offsets[codebook[ch]];

        /* Unsign if needed. */
        if (codebook[ch] == -1 || codebook[ch] == 2)
            unsign[ch] = 1 << (lsb_bits[ch] - 1);
        else
            unsign[ch] = 0;
    }

    for (i = 0; i < dp->blocksize; i++) {
        for (ch = rh->min_channel; ch <= rh->max_channel; ch++) {
            int32_t sample = ctx->sample_buffer[i][ch] >> 8;

            sample -= offset[ch];
            sample += unsign[ch];

            if (codebook[ch] >= 0) {
                int8_t vlc = (sample >> lsb_bits[ch]) + codebook_offset[ch];
                put_bits(pb, huffman_bits [codebook[ch]][vlc],
                             huffman_codes[codebook[ch]][vlc]);
            }

            put_sbits(pb, lsb_bits[ch], sample);
        }
    }
}

static int compare_filter_params(FilterParams *prev, FilterParams *fp)
{
    int i;

    if (prev->order != fp->order)
        return 1;
    if (prev->shift != fp->shift)
        return 1;

    for (i = 0; i < fp->order; i++)
        if (prev->coeff[i] != fp->coeff[i])
            return 1;

    return 0;
}

static int decoding_params_diff(MLPEncodeContext *ctx, DecodingParams *prev,
                                FilterParams filter_params[MAX_CHANNELS][NUM_FILTERS],
                                unsigned int substr, int write_all)
{
    DecodingParams *dp = &ctx->decoding_params[substr];
    RestartHeader  *rh = &ctx->restart_header [substr];
    unsigned int ch;
    int retval = 0;

    if (write_all)
        return PARAM_PRESENCE_FLAGS | PARAMS_DEFAULT;

    if (prev->param_presence_flags != dp->param_presence_flags)
        retval |= PARAM_PRESENCE_FLAGS;

    if (prev->blocksize != dp->blocksize)
        retval |= PARAM_BLOCKSIZE;

    if (prev->num_primitive_matrices != dp->num_primitive_matrices)
        retval |= PARAM_MATRIX;

    for (ch = 0; ch <= rh->max_matrix_channel; ch++)
        if (prev->output_shift[ch] != dp->output_shift[ch])
            retval |= PARAM_OUTSHIFT;

    for (ch = 0; ch <= rh->max_channel; ch++)
        if (prev->quant_step_size[ch] != dp->quant_step_size[ch])
            retval |= PARAM_QUANTSTEP;

    for (ch = rh->min_channel; ch <= rh->max_channel; ch++) {
        FilterParams *prev_fir = &filter_params[ch][FIR];
        FilterParams *prev_iir = &filter_params[ch][IIR];
        FilterParams *fir = &ctx->filter_params[ch][FIR];
        FilterParams *iir = &ctx->filter_params[ch][IIR];

        if (compare_filter_params(prev_fir, fir))
            retval |= PARAM_FIR;

        if (compare_filter_params(prev_iir, iir))
            retval |= PARAM_IIR;

        if (prev->huff_offset[ch] != dp->huff_offset[ch])
            retval |= PARAM_HUFFOFFSET;

        if (prev->codebook [ch] != dp->codebook [ch] ||
            prev->huff_lsbs[ch] != dp->huff_lsbs[ch])
            retval |= 0x1;
    }

    return retval;
}

static int mlp_encode_frame(AVCodecContext *avctx, uint8_t *buf, int buf_size,
                            void *data)
{
    FilterParams filter_params[MAX_CHANNELS][NUM_FILTERS];
    DecodingParams decoding_params[MAX_SUBSTREAMS];
    uint16_t substream_data_len[MAX_SUBSTREAMS];
    int32_t lossless_check_data[MAX_SUBSTREAMS];
    MLPEncodeContext *ctx = avctx->priv_data;
    uint8_t *buf2, *buf1, *buf0 = buf;
    uint16_t access_unit_header = 0;
    uint16_t parity_nibble = 0;
    int length, total_length;
    unsigned int substr;
    int channel, filter;
    int write_headers;
    int end = 0;

    if (avctx->frame_size > MAX_BLOCKSIZE) {
        av_log(avctx, AV_LOG_ERROR, "Invalid frame size (%d > %d)\n",
               avctx->frame_size, MAX_BLOCKSIZE);
        return -1;
    }

    memcpy(decoding_params, ctx->decoding_params, sizeof(decoding_params));
    memcpy(filter_params, ctx->filter_params, sizeof(filter_params));

    if (buf_size < 4)
        return -1;

    /* Frame header will be written at the end. */
    buf      += 4;
    buf_size -= 4;

    write_headers = !(avctx->frame_number & (MAJOR_HEADER_INTERVAL - 1));

    if (write_headers) {
        if (buf_size < 28)
            return -1;
        write_major_sync(ctx, buf, buf_size);
        buf      += 28;
        buf_size -= 28;
    }

    buf1 = buf;

    /* Substream headers will be written at the end. */
    for (substr = 0; substr < ctx->num_substreams; substr++) {
        buf      += 2;
        buf_size -= 2;
    }

    buf2 = buf;

    total_length = buf - buf0;

    input_data(ctx, data, lossless_check_data);

    for (channel = 0; channel < avctx->channels; channel++) {
        for (filter = 0; filter < NUM_FILTERS; filter++)
            set_filter_params(ctx, channel, filter);
        apply_filter(ctx, channel);
    }

    for (substr = 0; substr < ctx->num_substreams; substr++) {
        DecodingParams *dp = &ctx->decoding_params[substr];
        RestartHeader  *rh = &ctx->restart_header [substr];
        uint8_t parity, checksum;
        PutBitContext pb, tmpb;
        int params_changed;
        int last_block = 0;

        if (avctx->frame_size < dp->blocksize) {
            dp->blocksize = avctx->frame_size;
            last_block = 1;
        }

        determine_bits(ctx, substr);

        params_changed = decoding_params_diff(ctx, &decoding_params[substr],
                                              filter_params,
                                              substr, write_headers);

        init_put_bits(&pb, buf, buf_size);

        if (write_headers || params_changed) {
            put_bits(&pb, 1, 1);

            if (write_headers) {
                put_bits(&pb, 1, 1);

                write_restart_header(ctx, &pb, substr);
                rh->lossless_check_data = 0;
            } else {
                put_bits(&pb, 1, 0);
            }

            write_decoding_params(ctx, &pb, substr, params_changed);
        } else {
            put_bits(&pb, 1, 0);
        }

        rh->lossless_check_data ^= lossless_check_data[substr];

        write_block_data(ctx, &pb, substr);

        put_bits(&pb, 1, 1); /* TODO ??? */

        put_bits(&pb, (-put_bits_count(&pb)) & 15, 0);

        if (last_block) {
            /* TODO find a sample and implement shorten_by. */
            put_bits(&pb, 32, 0xd234d234);
        }

        /* data must be flushed for the checksum and parity to be right. */
        tmpb = pb;
        flush_put_bits(&tmpb);

        parity   = calculate_parity(buf, put_bits_count(&pb) >> 3) ^ 0xa9;
        checksum = mlp_checksum8   (buf, put_bits_count(&pb) >> 3);

        put_bits(&pb, 8, parity  );
        put_bits(&pb, 8, checksum);

        flush_put_bits(&pb);

        end += put_bits_count(&pb) >> 3;
        substream_data_len[substr] = end;

        buf += put_bits_count(&pb) >> 3;
    }

    length = buf - buf2;
    total_length += length;

    /* Write headers. */
    length = total_length / 2;

    parity_nibble  = ctx->timestamp;
    parity_nibble ^= length;

    for (substr = 0; substr < ctx->num_substreams; substr++) {
        uint16_t substr_hdr = 0;

        substr_hdr |= (0 << 15); /* extraword */
        substr_hdr |= (0 << 14); /* ??? */
        substr_hdr |= (1 << 13); /* checkdata */
        substr_hdr |= (0 << 12); /* ??? */
        substr_hdr |= (substream_data_len[substr] / 2) & 0x0FFF;

        AV_WB16(buf1, substr_hdr);

        parity_nibble ^= *buf1++;
        parity_nibble ^= *buf1++;
    }

    parity_nibble ^= parity_nibble >> 8;
    parity_nibble ^= parity_nibble >> 4;
    parity_nibble &= 0xF;

    access_unit_header |= (parity_nibble ^ 0xF) << 12;
    access_unit_header |= length & 0xFFF;

    AV_WB16(buf0  , access_unit_header);
    AV_WB16(buf0+2, ctx->timestamp    );

    ctx->timestamp += avctx->frame_size;

    return total_length;
}

static av_cold int mlp_encode_close(AVCodecContext *avctx)
{
    av_freep(&avctx->coded_frame);

    return 0;
}

AVCodec mlp_encoder = {
    "mlp",
    CODEC_TYPE_AUDIO,
    CODEC_ID_MLP,
    sizeof(MLPEncodeContext),
    mlp_encode_init,
    mlp_encode_frame,
    mlp_encode_close,
    .capabilities = CODEC_CAP_SMALL_LAST_FRAME,
    .long_name = NULL_IF_CONFIG_SMALL("Meridian Lossless Packing"),
};
