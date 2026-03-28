/*
 * Side-by-side 3D subtitle overlay filter with per-frame depth offset.
 *
 * Takes a 3840x1080 SBS YUV420P video and a 1920x1080 RGBA subtitle
 * (from sub2video), composites the subtitle onto both eye halves with
 * a horizontal pixel offset for 3D depth.
 *
 * The per-frame offset is read from AV_FRAME_DATA_SEI_UNREGISTERED side
 * data on the video frame (UUID prefix "OFMD"). The 3d-plane index is
 * read from the subtitle frame's metadata or overridden via option.
 *
 * Copyright (C) 2025
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation; either version 2.1 of the License,
 * or (at your option) any later version.
 */

#include "libavutil/opt.h"
#include "libavutil/imgutils.h"
#include "libavutil/pixdesc.h"
#include "avfilter.h"
#include "filters.h"
#include "formats.h"
#include "framesync.h"

typedef struct OverlaySBSContext {
    const AVClass *class;
    FFFrameSync fs;

    int plane;              /* OFMD plane index override (-1 = auto from metadata) */
    int fallback_offset;    /* offset when no OFMD side data present */

    int eye_width;          /* width of one eye (main_w / 2) */
    int eye_height;
    int hsub, vsub;         /* chroma subsampling of main video */

    int auto_plane;         /* plane index detected from subtitle metadata */
    int auto_plane_set;     /* whether auto_plane has been set */
} OverlaySBSContext;

#define OFFSET(x) offsetof(OverlaySBSContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM

static const AVOption overlay_sbs_options[] = {
    { "plane", "OFMD plane index (-1 = auto from 3d-plane metadata)",
      OFFSET(plane), AV_OPT_TYPE_INT, { .i64 = -1 }, -1, 31, FLAGS },
    { "fallback_offset", "pixel offset when no OFMD data present",
      OFFSET(fallback_offset), AV_OPT_TYPE_INT, { .i64 = 0 }, -127, 127, FLAGS },
    { "eof_action", "action on subtitle EOF",
      OFFSET(fs.opt_eof_action), AV_OPT_TYPE_INT, { .i64 = EOF_ACTION_PASS },
      EOF_ACTION_REPEAT, EOF_ACTION_PASS, FLAGS, .unit = "eof_action" },
        { "repeat", "Repeat the previous frame",   0, AV_OPT_TYPE_CONST, { .i64 = EOF_ACTION_REPEAT }, .flags = FLAGS, .unit = "eof_action" },
        { "endall", "End both streams",             0, AV_OPT_TYPE_CONST, { .i64 = EOF_ACTION_ENDALL }, .flags = FLAGS, .unit = "eof_action" },
        { "pass",   "Pass through the main input",  0, AV_OPT_TYPE_CONST, { .i64 = EOF_ACTION_PASS },   .flags = FLAGS, .unit = "eof_action" },
    { "shortest", "force shortest", OFFSET(fs.opt_shortest),
      AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, FLAGS },
    { NULL }
};

FRAMESYNC_DEFINE_CLASS(overlay_sbs, OverlaySBSContext, fs);

/* ---- Read OFMD offset from video frame side data ---- */

static int get_ofmd_offset(const AVFrame *video, int plane_idx)
{
    int i;
    for (i = 0; i < video->nb_side_data; i++) {
        const AVFrameSideData *sd = video->side_data[i];
        if (sd->type == AV_FRAME_DATA_SEI_UNREGISTERED &&
            sd->size >= 18 &&
            sd->data[0] == 'O' && sd->data[1] == 'F' &&
            sd->data[2] == 'M' && sd->data[3] == 'D') {
            int num_planes = sd->data[16];
            int swap_eyes  = sd->data[17];
            if (plane_idx >= 0 && plane_idx < num_planes &&
                sd->size >= 18 + num_planes) {
                int off = (int)(int8_t)sd->data[18 + plane_idx];
                return swap_eyes ? -off : off;
            }
            return 0;
        }
    }
    return 0;  /* no OFMD data */
}

/* ---- Composite RGBA subtitle onto YUV420P SBS frame ---- */

static void blend_subtitle_onto_eye(AVFrame *dst, const AVFrame *sub,
                                    int eye_x_offset, int h_offset,
                                    int eye_width, int eye_height,
                                    int hsub, int vsub)
{
    int x, y;
    int sub_w = sub->width;
    int sub_h = sub->height;
    const uint8_t *src_row;
    uint8_t *dst_y, *dst_u, *dst_v;
    int dst_stride_y = dst->linesize[0];
    int dst_stride_u = dst->linesize[1];
    int dst_stride_v = dst->linesize[2];
    int src_stride   = sub->linesize[0];

    for (y = 0; y < sub_h && y < eye_height; y++) {
        src_row = sub->data[0] + y * src_stride;
        dst_y = dst->data[0] + y * dst_stride_y + eye_x_offset;

        for (x = 0; x < sub_w; x++) {
            /* RGBA pixel from subtitle */
            uint8_t r = src_row[x * 4 + 0];
            uint8_t g = src_row[x * 4 + 1];
            uint8_t b = src_row[x * 4 + 2];
            uint8_t a = src_row[x * 4 + 3];

            if (a == 0)
                continue;

            /* Apply horizontal offset */
            int dx = x + h_offset;
            if (dx < 0 || dx >= eye_width)
                continue;

            /* Convert RGB to YUV (BT.709) */
            int Y = (( 47 * r + 157 * g +  16 * b + 128) >> 8) + 16;
            int U = ((-26 * r -  87 * g + 112 * b + 128) >> 8) + 128;
            int V = ((112 * r -  102 * g -  10 * b + 128) >> 8) + 128;

            /* Alpha blend Y plane */
            int dst_idx = eye_x_offset + dx;
            uint8_t *py = dst->data[0] + y * dst_stride_y + dst_idx;
            *py = (uint8_t)((*py * (255 - a) + Y * a + 127) / 255);

            /* Alpha blend U/V planes (chroma subsampled) */
            if ((y & ((1 << vsub) - 1)) == 0 && (dx & ((1 << hsub) - 1)) == 0) {
                int cx = (dst_idx) >> hsub;
                int cy = y >> vsub;
                uint8_t *pu = dst->data[1] + cy * dst_stride_u + cx;
                uint8_t *pv = dst->data[2] + cy * dst_stride_v + cx;
                *pu = (uint8_t)((*pu * (255 - a) + U * a + 127) / 255);
                *pv = (uint8_t)((*pv * (255 - a) + V * a + 127) / 255);
            }
        }
    }
}

/* ---- Framesync callback ---- */

static int process_frame(FFFrameSync *fs)
{
    AVFilterContext *ctx = fs->parent;
    OverlaySBSContext *s = ctx->priv;
    AVFrame *video, *sub;
    int ret, offset, plane_idx;

    ret = ff_framesync_dualinput_get_writable(fs, &video, &sub);
    if (ret < 0)
        return ret;

    if (!sub)
        return ff_filter_frame(ctx->outputs[0], video);

    /* Determine plane index */
    plane_idx = s->plane;
    if (plane_idx < 0) {
        /* Auto-detect from subtitle frame metadata */
        if (!s->auto_plane_set) {
            const AVDictionaryEntry *e = av_dict_get(sub->metadata, "3d-plane", NULL, 0);
            if (e) {
                s->auto_plane = atoi(e->value);
                av_log(ctx, AV_LOG_INFO, "overlay_sbs: auto-detected 3d-plane=%d\n", s->auto_plane);
            } else {
                s->auto_plane = 0;
                av_log(ctx, AV_LOG_WARNING, "overlay_sbs: no 3d-plane metadata, defaulting to plane 0\n");
            }
            s->auto_plane_set = 1;
        }
        plane_idx = s->auto_plane;
    }

    /* Get per-frame offset from OFMD side data */
    offset = get_ofmd_offset(video, plane_idx);
    av_log(ctx, AV_LOG_DEBUG, "overlay_sbs: frame pts=%"PRId64" plane=%d ofmd_offset=%d sub=%dx%d\n",
           video->pts, plane_idx, offset, sub->width, sub->height);
    if (offset == 0)
        offset = s->fallback_offset;

    /* Composite subtitle onto left eye (x=0, no shift) */
    blend_subtitle_onto_eye(video, sub, 0, 0,
                            s->eye_width, s->eye_height,
                            s->hsub, s->vsub);

    /* Composite subtitle onto right eye (x=eye_width, with depth offset) */
    blend_subtitle_onto_eye(video, sub, s->eye_width, offset,
                            s->eye_width, s->eye_height,
                            s->hsub, s->vsub);

    return ff_filter_frame(ctx->outputs[0], video);
}

/* ---- Filter lifecycle ---- */

static av_cold int overlay_sbs_init(AVFilterContext *ctx)
{
    OverlaySBSContext *s = ctx->priv;
    s->fs.on_event = process_frame;
    return 0;
}

static av_cold void overlay_sbs_uninit(AVFilterContext *ctx)
{
    OverlaySBSContext *s = ctx->priv;
    ff_framesync_uninit(&s->fs);
}

static int config_input_main(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    OverlaySBSContext *s = ctx->priv;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);

    s->hsub = desc->log2_chroma_w;
    s->vsub = desc->log2_chroma_h;
    s->eye_width = inlink->w / 2;
    s->eye_height = inlink->h;

    av_log(ctx, AV_LOG_INFO, "overlay_sbs: SBS video %dx%d, eye=%dx%d\n",
           inlink->w, inlink->h, s->eye_width, s->eye_height);
    return 0;
}

static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    OverlaySBSContext *s = ctx->priv;
    int ret;

    if ((ret = ff_framesync_init_dualinput(&s->fs, ctx)) < 0)
        return ret;

    outlink->w = ctx->inputs[0]->w;
    outlink->h = ctx->inputs[0]->h;
    outlink->time_base = ctx->inputs[0]->time_base;

    return ff_framesync_configure(&s->fs);
}

static int overlay_sbs_activate(AVFilterContext *ctx)
{
    OverlaySBSContext *s = ctx->priv;
    return ff_framesync_activate(&s->fs);
}

static int query_formats(const AVFilterContext *ctx,
                         AVFilterFormatsConfig **cfg_in,
                         AVFilterFormatsConfig **cfg_out)
{
    /* Main input: YUV420P */
    static const enum AVPixelFormat main_fmts[] = {
        AV_PIX_FMT_YUV420P, AV_PIX_FMT_NONE
    };
    /* Subtitle input: RGBA (from sub2video) */
    static const enum AVPixelFormat sub_fmts[] = {
        AV_PIX_FMT_RGBA, AV_PIX_FMT_NONE
    };
    /* Output: YUV420P */
    int ret;

    if ((ret = ff_formats_ref(ff_make_format_list(main_fmts),
                              &cfg_in[0]->formats)) < 0)
        return ret;
    if ((ret = ff_formats_ref(ff_make_format_list(sub_fmts),
                              &cfg_in[1]->formats)) < 0)
        return ret;
    if ((ret = ff_formats_ref(ff_make_format_list(main_fmts),
                              &cfg_out[0]->formats)) < 0)
        return ret;
    return 0;
}

/* ---- Pads ---- */

static const AVFilterPad overlay_sbs_inputs[] = {
    {
        .name         = "main",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_input_main,
    },
    {
        .name         = "overlay",
        .type         = AVMEDIA_TYPE_VIDEO,
    },
};

static const AVFilterPad overlay_sbs_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
    },
};

const AVFilter ff_vf_overlay_sbs = {
    .name           = "overlay_sbs",
    .description    = NULL_IF_CONFIG_SMALL("Overlay subtitles on side-by-side 3D video with depth offset."),
    .preinit        = overlay_sbs_framesync_preinit,
    .init           = overlay_sbs_init,
    .uninit         = overlay_sbs_uninit,
    .priv_size      = sizeof(OverlaySBSContext),
    .priv_class     = &overlay_sbs_class,
    .activate       = overlay_sbs_activate,
    FILTER_INPUTS(overlay_sbs_inputs),
    FILTER_OUTPUTS(overlay_sbs_outputs),
    FILTER_QUERY_FUNC2(query_formats),
    .flags          = AVFILTER_FLAG_SUPPORT_TIMELINE_INTERNAL,
};
