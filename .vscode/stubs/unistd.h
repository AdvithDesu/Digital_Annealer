#pragma once
// Stub for MSVC IntelliSense — POSIX unistd.h
#include <io.h>
#include <process.h>
typedef int pid_t;
int getopt(int argc, char* const argv[], const char* optstring);
extern char* optarg;
extern int optind, opterr, optopt;
