#ifndef WAVE_EXPLORER_H
#define WAVE_EXPLORER_H

#include <iostream>
#include <vector>
#include <math.h>

#include <cairo/cairo.h>

//#define NB_SAMPLES_PER_FRAME 	1152
//#define NB_SAMPLES_PER_FRAME 	9216
//#define NB_SAMPLES_PER_FRAME 	18432
#define NB_SAMPLES_PER_FRAME 	44100

extern "C" 
{
	#include "libavcodec/avcodec.h"
	#include "libavformat/avformat.h"
	#include "libavutil/avutil.h"
    //#include "libavutil/avstring.h"
	//#include "libavutil/mathematics.h"
}


class WaveSequence{
public:
	int nb_waves;
	std::vector<double> real;
	std::vector<double> imag;
	std::vector<int> index;

	WaveSequence();
	void fast_fourier_transform(std::vector<int> samples, int nb_samples);
	void sort_waves();
};


class WaveExplorer{
private:
	// ffmpeg data-structure to decode mp3 file
	AVFormatContext *format_ctx;
    AVCodecContext *codec_ctx;
    AVCodec *codec;
    AVFrame *frame;
    AVPacket packet;

    bool empty_song;

    // decoded samples of a frame, and waves by fft
    int nb_samples;
    std::vector<int> samples;
    int next_sample_pos;
    
    WaveSequence waves;

	// alloc and set samples of a frame
	int get_next_frame_samples();

	// use cairo to draw frames
	void draw_samples(cairo_t *cr, int width);
	void draw_kwaves(cairo_t *cr, int width, int k);

public:

	WaveExplorer();
	~WaveExplorer();

	// open, find and load codec, return the duration
	int64_t load_file(const char *file_path);

	// parse a frame: decode, fft, return nb_samples
	const WaveSequence* get_next_frame_waves();

	// draw samples, k key waves
	void draw_frames(const char *file_path, int k);

	void close_file();

};

#endif