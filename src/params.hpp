/***************************************************************************

	Copyright (C) 2015-16 Tom Furnival

***************************************************************************/

#ifndef PARAMS_H
#define PARAMS_H

#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

void ParseParameters(std::istream & cfgfile, std::map<std::string, std::string>& options);

#endif
