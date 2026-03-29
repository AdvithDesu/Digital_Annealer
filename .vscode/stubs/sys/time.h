#pragma once
// Stub for MSVC IntelliSense — POSIX sys/time.h
#include <time.h>
struct timeval {
    long tv_sec;
    long tv_usec;
};
struct timezone {
    int tz_minuteswest;
    int tz_dsttime;
};
int gettimeofday(struct timeval* tv, struct timezone* tz);
