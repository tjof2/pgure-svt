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

void ParseParameters(std::istream & cfgfile, std::map<std::string, std::string>& options) {
    for (std::string line; std::getline(cfgfile, line); ) {
        std::istringstream iss(line);
        std::string id, eq, val, temp;

        if (!(iss >> id)) {
            continue;    // Ignore empty lines
        }
        else if (id[0] == '#') {
            continue;    // Ignore comment lines
        }
        else if (!(iss >> eq ) || eq != ":" || iss.get() != EOF) {
            while( iss >> temp ) {
                if( iss >> std::ws) {
                    val += temp;
                }
                else {
                    val += temp + " ";
                }
            }
        }

        // Set the parameter
        options[id] = val;
    }
    return;
}

#endif
