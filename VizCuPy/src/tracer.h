#pragma once

namespace ftxj {
namespace profiler {

class TracerBase {
    virtual ~TracerBase() = default;
    virtual void start() = 0;
    virtual void stop() = 0;
};

}
}