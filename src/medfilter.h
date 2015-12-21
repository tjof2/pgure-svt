#ifndef MEDFILTER_H
#define MEDFILTER_H

#ifdef __cplusplus
	extern "C" {
#endif

void ConstantTimeMedianFilter(
		const unsigned short* const src, unsigned short* const dst,
		int width, int height,
		int src_step_row, int dst_step_row,
		int r, int channels, unsigned long memsize
		);

#ifdef __cplusplus
	};
#endif


#endif
