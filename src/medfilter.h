/***************************************************************************

  Copyright (C) 2015-2020 Tom Furnival

  This file is part of  PGURE-SVT.

  PGURE-SVT is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  PGURE-SVT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PGURE-SVT. If not, see <http://www.gnu.org/licenses/>.

***************************************************************************/

#ifndef MEDFILTER_H
#define MEDFILTER_H

#ifdef __cplusplus
extern "C"
{
#endif

    void ConstantTimeMedianFilter(const unsigned short *const src,
                                  unsigned short *const dst, int width, int height,
                                  int src_step_row, int dst_step_row, int r,
                                  int channels, unsigned long memsize);

#ifdef __cplusplus
};
#endif

#endif
