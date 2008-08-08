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
    AVCodecContext *avctx;

    int             num_substreams;

    int             sample_fmt;
    int             sample_rate;

    int32_t         sample_buffer[MAX_BLOCKSIZE][MAX_CHANNELS+2];

    uint16_t        timestamp;

    uint8_t         mlp_channels;

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

/* TODO pass only PutBitContext and use pb->buffer. */
static void write_restart_header(MLPEncodeContext *ctx, uint8_t *buf,
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

    checksum = mlp_restart_checksum(buf, put_bits_count(pb) - start_count);

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
/*      param_presence_flags |= PARAM_FIR; */
/*      param_presence_flags |= PARAM_IIR; */
        param_presence_flags |= PARAM_HUFFOFFSET;

        dp->param_presence_flags = param_presence_flags;
    }

    return 0;
}

static void write_filter_params(MLPEncodeContext *ctx, PutBitContext *pb,
                                unsigned int channel, unsigned int filter)
{
    return;
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
                sample = (sample << 8) & 0x00ffffff;
                lossless_check_data_temp ^= sample << channel;
                ctx->sample_buffer[i][channel] = sample;
            }
        }

        lossless_check_data[substr] = lossless_check_data_temp;
    }
}

static const uint8_t huffman_tables[3][18][2] = {
    {    /* huffman table 0, -7 - +10 */
        {0x01, 9}, {0x01, 8}, {0x01, 7}, {0x01, 6}, {0x01, 5}, {0x01, 4}, {0x01, 3},
        {0x04, 3}, {0x05, 3}, {0x06, 3}, {0x07, 3},
        {0x03, 3}, {0x05, 4}, {0x09, 5}, {0x11, 6}, {0x21, 7}, {0x41, 8}, {0x81, 9},
    }, { /* huffman table 1, -7 - +8 */
        {0x01, 9}, {0x01, 8}, {0x01, 7}, {0x01, 6}, {0x01, 5}, {0x01, 4}, {0x01, 3},
        {0x02, 2}, {0x03, 2},
        {0x03, 3}, {0x05, 4}, {0x09, 5}, {0x11, 6}, {0x21, 7}, {0x41, 8}, {0x81, 9},
    }, { /* huffman table 2, -7 - +7 */
        {0x01, 9}, {0x01, 8}, {0x01, 7}, {0x01, 6}, {0x01, 5}, {0x01, 4}, {0x01, 3},
        {0x01, 1},
        {0x03, 3}, {0x05, 4}, {0x09, 5}, {0x11, 6}, {0x21, 7}, {0x41, 8}, {0x81, 9},
    }
};

static int codebook_extremes[3][2] = {
    {-9, 8}, {-8, 7}, {-7, 7},
};

static int codebook_offsets[3] = {
    9, 8, 7,
};

static int no_codebook_bits(MLPEncodeContext *ctx, unsigned int substr,
                            unsigned int channel,
                            int16_t min, int16_t max,
                            int16_t *poffset, int *plsb_bits)
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

    for (lsb_bits = 16; lsb_bits && !(diff & (1<<(lsb_bits-1))); lsb_bits--);

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

    *poffset   = offset;
    *plsb_bits = lsb_bits;

    return lsb_bits * dp->blocksize;
}

static void codebook_bits_offset(MLPEncodeContext *ctx, unsigned int substr,
                                 unsigned int channel, int codebook,
                                 int32_t min, int32_t max, int16_t offset,
                                 int *plsb_bits, int *pcount)
{
    DecodingParams *dp = &ctx->decoding_params[substr];
    int codebook_offset  = codebook_offsets[codebook];
    int32_t codebook_min = codebook_extremes[codebook][0];
    int32_t codebook_max = codebook_extremes[codebook][1];
    int lsb_bits = 0, bitcount = 0;
    int i;

    min -= offset;
    max -= offset;

    while (min < codebook_min || max > codebook_max) {
        lsb_bits++;
        min >>= 1;
        max >>= 1;
    }

    for (i = 0; i < dp->blocksize; i++) {
        int32_t sample = (int16_t) (ctx->sample_buffer[i][channel] >> 8);

        sample  -= offset;
        sample >>= lsb_bits;

        bitcount += huffman_tables[codebook][sample + codebook_offset][1];
    }

    if (codebook == 2)
        lsb_bits++;

    *plsb_bits = lsb_bits;
    *pcount    = lsb_bits * dp->blocksize + bitcount;
}

static int codebook_bits(MLPEncodeContext *ctx, unsigned int substr,
                         unsigned int channel, int codebook,
                         int16_t min, int16_t max,
                         int16_t *poffset, int *plsb_bits)
{
    int best_count = INT_MAX;
    int16_t best_offset = 0;
    int best_lsb_bits = 0;
    int offset;
    int offset_min, offset_max;

    offset_min = FFMAX(min, HUFF_OFFSET_MIN);
    offset_max = FFMIN(max, HUFF_OFFSET_MAX);

    for (offset = offset_min; offset <= offset_max; offset++) {
        int lsb_bits, count;

        codebook_bits_offset(ctx, substr, channel, codebook,
                             min, max, offset,
                             &lsb_bits, &count);

        if (count < best_count) {
            best_lsb_bits = lsb_bits;
            best_offset   = offset;
            best_count    = count;
         }
    }

    *plsb_bits = best_lsb_bits;
    *poffset   = best_offset;

    return best_count;
}

static void determine_bits(MLPEncodeContext *ctx)
{
    unsigned int substr;

    for (substr = 0; substr < ctx->num_substreams; substr++) {
        DecodingParams *dp = &ctx->decoding_params[substr];
        RestartHeader  *rh = &ctx->restart_header [substr];
        unsigned int channel;

        for (channel = 0; channel <= rh->max_channel; channel++) {
            int16_t min = INT16_MAX, max = INT16_MIN;
            int best_bitcount = INT_MAX;
            int best_codebook = 0;
            int16_t offset[3];
            int bitcount[3];
            int lsb_bits[3], i;

            /* Determine extremes. */
            for (i = 0; i < dp->blocksize; i++) {
                int16_t sample = ctx->sample_buffer[i][channel] >> 8;
                if (sample < min)
                    min = sample;
                if (sample > max)
                    max = sample;
            }

            bitcount[0] = no_codebook_bits(ctx, substr, channel,
                                           min, max, &offset[0], &lsb_bits[0]);

            for (i = 1; i < 3; i++) {
                bitcount[i] = codebook_bits(ctx, substr, channel, i - 1,
                                            min, max, &offset[i], &lsb_bits[i]);
            }

            /* Choose best codebook. */
            for (i = 0; i < 3; i++) {
                if (bitcount[i] < best_bitcount) {
                    best_bitcount = bitcount[i];
                    best_codebook = i;
                }
            }

            /* Update context. */
            dp->huff_offset[channel] = offset  [best_codebook];
            dp->huff_lsbs  [channel] = lsb_bits[best_codebook] + 8;
            dp->codebook   [channel] = best_codebook;
        }
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
        lsb_bits[ch] = dp->huff_lsbs[ch] - dp->quant_step_size[ch];
        codebook       [ch] = dp->codebook   [ch] - 1;
        offset  [ch] = dp->huff_offset[ch];
        codebook_offset[ch] = codebook_offsets[codebook[ch]];

        /* Unsign if needed. */
        if (codebook[ch] == -1 || codebook[ch] == 2)
        unsign  [ch] = 1 << (lsb_bits[ch] - 1);
        else
            unsign[ch] = 0;
    }

    for (i = 0; i < dp->blocksize; i++) {
        for (ch = rh->min_channel; ch <= rh->max_channel; ch++) {
            int32_t sample = (int16_t) (ctx->sample_buffer[i][ch] >> 8);

            sample -= offset[ch];
            sample += unsign[ch];

            if (codebook[ch] >= 0) {
                int8_t vlc = (sample >> lsb_bits[ch]) + codebook_offset[ch];
                put_bits(pb, huffman_tables[codebook[ch]][vlc][1],
                             huffman_tables[codebook[ch]][vlc][0]);
            }

            put_sbits(pb, lsb_bits[ch], sample);
        }
    }
}

static int decoding_params_diff(MLPEncodeContext *ctx, DecodingParams *prev,
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

        /* TODO Check filters. */

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
    DecodingParams decoding_params[MAX_SUBSTREAMS];
    uint16_t substream_data_len[MAX_SUBSTREAMS];
    int32_t lossless_check_data[MAX_SUBSTREAMS];
    MLPEncodeContext *ctx = avctx->priv_data;
    uint8_t *buf2, *buf1, *buf0 = buf;
    uint16_t access_unit_header = 0;
    uint16_t parity_nibble = 0;
    int length, total_length;
    unsigned int substr;
    int write_headers;
    PutBitContext pb;
    int end = 0;

    if (avctx->frame_size > MAX_BLOCKSIZE) {
        av_log(avctx, AV_LOG_ERROR, "Invalid frame size (%d > %d)\n",
               avctx->frame_size, MAX_BLOCKSIZE);
        return -1;
    }

    memcpy(decoding_params, ctx->decoding_params, sizeof(decoding_params));

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

    determine_bits(ctx);

    for (substr = 0; substr < ctx->num_substreams; substr++) {
        DecodingParams *dp = &ctx->decoding_params[substr];
        RestartHeader  *rh = &ctx->restart_header [substr];
        uint8_t parity, checksum;
        PutBitContext tmpb;
        int params_changed;
        int last_block = 0;

        init_put_bits(&pb, buf, buf_size);

        if (avctx->frame_size < dp->blocksize) {
            dp->blocksize = avctx->frame_size;
            last_block = 1;
        }

        params_changed = decoding_params_diff(ctx, &decoding_params[substr],
                                              substr, write_headers);

        if (write_headers || params_changed) {
            put_bits(&pb, 1, 1);

            if (write_headers) {
                put_bits(&pb, 1, 1);

                write_restart_header(ctx, buf, &pb, substr);
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
