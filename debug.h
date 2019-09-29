#ifndef DEBUG_H
#define DEBUG_H

#include <iostream>

#define DEBUG

#ifdef DEBUG

#define DBG(a) { \
    std::cout<<"file:"<<__FILE__<<"  line:"<<__LINE__<<std::endl; \
    std::cout<<a<<std::endl; \
}

#else
#define DBG(a) {}
#endif

#endif // DEBUG_H
