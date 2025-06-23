#include <iostream>

#define C_BASED 0
#if C_BASED
#include <sys/times.h>
#include <unistd.h>

class Timer {
  public:
    Timer() { restart(); }

    void restart() {
      tms st;
      wall_ = times(&st);
      user_ = st.tms_utime + st.tms_cutime;
      sys_ = st.tms_stime + st.tms_cstime;
    }

    void report(const std::string& msg = "PERF: ") {
      std::cout << msg <<
        getUserSecs() << "u " <<
        getSysSecs() << "s " <<
        getWallSecs() << "w\n";
    }

  private:
    double getUserSecs() {
      tms now;
      times(&now);
      long user = now.tms_utime + now.tms_cutime - user_;
      return (double) user / sysconf(_SC_CLK_TCK);
    }

    double getSysSecs() {
      tms now;
      times(&now);
      long sys = now.tms_stime + now.tms_cstime - sys_;
      return (double) sys / sysconf(_SC_CLK_TCK);
    }

    double getWallSecs() {
      tms now;
      long wall = times(&now) - wall_;
      return (double) wall / sysconf(_SC_CLK_TCK);
    }

  private:
    long user_;
    long sys_;
    long wall_;
};

#else

#include <chrono>

class Timer {
  public:
    Timer() : start_(std::chrono::steady_clock::now()) {}

    void restart() {
      start_ = std::chrono::steady_clock::now();
    }

    void report(const std::string& msg = "PERF: ") {
      const auto end = std::chrono::steady_clock::now();
      const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
      std::cout << msg << duration << " ms\n";
    }

    static void print_current_time(const std::string& msg = "PERF: ") {
      auto now = std::chrono::steady_clock::now();
      auto time_since_epoch = now.time_since_epoch();
      auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(time_since_epoch);
      std::cout << msg << milliseconds.count() << std::endl;
    }

  private:
    std::chrono::time_point<std::chrono::steady_clock> start_;
};
#endif
