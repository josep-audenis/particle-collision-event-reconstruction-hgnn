#include "helper.h"
#include <iostream>
#include <getopt.h>


void printUsage(const char *program) {
    std::cout << "Usage: " << program << " [OPTIONS]\n"
            << "Options:\n"
            << "  -d, --decay DECAY    Name parameter (required)\n"
            << "  -s, --start VALUE    Start value (required)\n"
            << "  -e, --end VALUE      End value (required)\n"
            << "  -i, --indir INDIR    Input directory (optional)\n"
            << "  -b, --bunch SIZE     Number of events per bunch (optional)\n"
            << "  -m, --mode MODE      Reasso using true or associated PV info (optional)\n"
            << "\nExample:\n"
            << "  " << program << " -n Bs_Jpsiphi -s 0 -e 100\n"
            << "  " << program << " --name=Bs_Jpsiphi --start=0 --end=100\n";
}


Options parseOptions(int argc, char **argv) {
    Options opts;

    // Define long options
    static struct option long_options[] = {
        {"decay", required_argument, 0, 'd'},
        {"start", required_argument, 0, 's'},
        {"end", required_argument, 0, 'e'},
        {"indir", optional_argument, 0, 'i'},
        {"bunch", optional_argument, 0, 'b'},
        {"mode", optional_argument, 0, 'm'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;

    while ((opt = getopt_long(argc, argv, "d:s:e:i:b:m:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'd':
                opts.decay = optarg;
                break;
            case 's':
                try {
                    opts.start = std::stoi(optarg);
                } catch (...) {
                    std::cerr << "Error: Invalid start value '" << optarg << "'\n";
                    return opts;
                }
                break;
            case 'e':
                try {
                    opts.end = std::stoi(optarg);
                } catch (...) {
                    std::cerr << "Error: Invalid end value '" << optarg << "'\n";
                    return opts;
                }
                break;
            case 'i':
                opts.indir = optarg;
                if (opts.indir.back() != '/') {
                    opts.indir += '/';
                }
                break;
            case 'b':
                try {
                    opts.bunch_size = std::stoi(optarg);
                } catch (...) {
                    std::cerr << "Error: Invalid indir value '" << optarg << "'\n";
                    return opts;
                }
                break;
            case 'm':
                opts.mode = optarg;
                break;
            case '?':
                // getopt_long already printed an error message
                return opts;
            default:
                return opts;
        }
    }

    if (opts.decay.empty()) {
        std::cerr << "Error: --decay is required\n\n";
        printUsage(argv[0]);
        return opts;
    }

    if (opts.start == -1) {
        std::cerr << "Error: --start is required\n\n";
        printUsage(argv[0]);
        return opts;
    }

    if (opts.end == -1) {
        std::cerr << "Error: --end is required\n\n";
        printUsage(argv[0]);
        return opts;
    }

    opts.indir = opts.indir + opts.decay + "/";
    opts.valid = true;
    return opts;
}
