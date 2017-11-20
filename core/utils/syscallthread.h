#include <iostream>
// Copyright (c) 2016-2017, Nefeli Networks, Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// * Neither the names of the copyright holders nor the names of their
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef BESS_UTILS_SYSCALLTHREAD_H
#define BESS_UTILS_SYSCALLTHREAD_H

#include <cassert>
#include <cerrno>
#include <thread>

#include <nmmintrin.h>
#include <signal.h>

namespace bess {
namespace utils {

template <bool>
class SyscallThread;  // forward

bool CatchExitSignal();

/*!
 * At several points in bess we need to spin off threads that make
 * blocking system calls, and be able to tell these threads to
 * terminate.
 *
 * The actual code inside these threads varies a lot, but there
 * is some common setup:
 *
 *  - establish a signal handler for the signal so that
 *    the signal will interrupt blocking system calls
 *  - establish reception of that signal in the thread
 *  - send such a signal from outside the thread
 *
 * Signal handlers are themselves global across all threads, so
 * the signal-handler-establishing can be done just once.
 *
 * We also pick which signal is to be used to interrupt the
 * blocking system call, and provide a function to send that
 * signal.
 */

class ExitSigMask {
 public:
  ExitSigMask() {
    sigfillset(&allmask_);  // complete blockage of everything
    sigfillset(&sigmask_);  // block all except SIG_THREAD_EXIT
    sigdelset(&sigmask_, SIG_THREAD_EXIT);
  }
  const sigset_t &GetAllMask() const { return allmask_; }
  const sigset_t &GetExitMask() const { return sigmask_; }

 private:
  static constexpr int SIG_THREAD_EXIT = SIGUSR2;
  friend SyscallThread<true>;
  friend SyscallThread<false>;
  friend bool CatchExitSignal();
  sigset_t allmask_;
  sigset_t sigmask_;

  // We only catch the exit signal once, process wide, so we want
  // a flag to tell if we've done that.  Likewise, the per-thread
  // signal mask is the same across all such threads, so we only
  // need one instance.
  static bool signals_initialized_;
};

// pthread_sigmask shouldn't return EINTR, but we can check.
static inline void ThreadSetSigmask(const sigset_t *mask) {
  while (pthread_sigmask(SIG_SETMASK, mask, nullptr) < 0 && errno == EINTR) {
    continue;
  }
}

/*!
 * You can use this class somewhat like std::thread, but it's a
 * bit more like Python's thread class.  Instead of t = Thread(f),
 * you define a Run() function in your derived class.  Here's a
 * realish example:
 *
 *   class SomeThread : public SyscallThread<bool Reliable> {
 *     ...
 *     void Run() { ... code goes here ... }
 *   };
 *   class X {
 *     ...
 *     SomeThread thread;
 *     ...
 *   };
 *   X var;
 *   ...
 *   var.thread.Start();
 *
 * and your SomeThread::Run() will be called in the started thread.
 *
 * In Run(), check IsExitRequested() any time your system call(s)
 * return(s) with an EINTR error, as these are requests for your
 * thread to terminate.  You can check it any other time as well.
 * (We check it once before even calling Run(), so if you do
 *
 * Optionally (for performance / avoiding more signals), you may
 * call BeginExiting() once you are on the way out, but have not
 * actually finished.  This tells the knock thread (see below) that
 * its job is done.
 *
 * Once your Run() returns (the thread has exited), var.thread.Done()
 * will return True.
 *
 * You may WaitFor() the thread, or call Terminate(), at any time:
 * these are no-ops if the thread was never Start()ed.
 *
 * Once Start()ed, the thread is not re-Start()able until
 * Terminate()d and/or WaitFor()ed, after which you may --
 * CAREFULLY (e.g., under locks if this could race) -- invoke
 * Reset() to put it back to "never started" state.
 *
 * Because many system calls are not available in a reliable-signal
 * flavor (cf. pselect/ppoll), requesting an exit normally starts a
 * "knock thread" that keeps kicking your Run() code to get it to
 * return.
 *
 * This part is optional: if you declare that you are using the
 * reliable signal system call, and never block in any other syscall,
 * you can skip it.  In this case you should call Sigmask() to
 * obtain the correct mask to use in pselect/ppol, and then you
 * *must* check IsExitRequested() immediately after the pselect/ppoll
 * returns.
 *
 * Otherwise, there is a race between any IsExitRequested() test and
 * the entry to a system call.  This is why we have a knock thread:
 * it will repeatedly send the interrupt signal.  At least one of
 * these will cause the system call itself to be interrupted and
 * return -1 with errno set to EINTR.
 *
 * This means that when using SyscallThread<false>, if you need to
 * make system calls in Run() that must *not* be interrupted, you
 * should call PushDefer() first.  Knock-thread signals will be
 * deferred until a corresponding PopDefer().
 *
 * TODO(torek): attempt to template away knock_thread_ related code
 * when using reliable signals.
 *
 * MAYBE-to-do(torek): allow detaching (move the state variables
 * into a sub-object that is given to the thread, so that we can
 * use thread_.detach() and knock_thread_.detach()).  If we do this
 * we need new/dispose, std::unique_ptr<...>, and/or refcounts.
 */
template <bool ReliableSignals>
class SyscallThread {
 public:
  // NB: order matters here.  Thread state progresses linearly
  // (except for Reset() which must not be allowed to race).
  enum ThreadState { TSNotStarted, TSStarting, TSReady, TSExiting, TSDone };

  // to allow a future Detach, we have a WaitType enumeration
  enum WaitType { RequestOnly, Wait };

  SyscallThread()
      : state_(TSNotStarted),
        exit_requested_(false),
        thread_(),
        knock_thread_(),
        defer_count_(0) {}

  /*!
   * Re-set state to allow re-firing thread.  USE WITH CAUTION!
   * Returns false if you called it inappropriately, true if it
   * did the reset.
   *
   * (If we ever allow detaching threads, this code will be
   * responsible for allocating new state objects if needed.)
   */
  bool Reset() {
    if (state_ == TSNotStarted) {
      // Nothing to do.
      return true;
    }

    if (state_ != TSDone) {
      // Inappropriate call.
      return false;
    }

    // Collect previous threads, if we have not yet.
    WaitFor();

    // Rewind state.
    exit_requested_ = false;
    state_ = TSNotStarted;
    if (ReliableSignals) {
      defer_count_ = 0;
    }
    return true;
  }

  /*!
   * Start the thread running.  It will call the user provided Run()
   * once it's ready (unless the thread was already asked to exit).
   *
   * Returns true on success, false (with errno set) on failure.
   */
  bool Start() {
    // Establish process-wide signal catching, if that isn't
    // yet done.  The signal handler will interrupt system calls.
    if (!ExitSigMask::signals_initialized_) {
      if (!CatchExitSignal()) {
        return false;
      }
    }

    // If the thread isn't in pristine state, this is an error.
    if (state_ != TSNotStarted) {
      errno = EINVAL;
      return false;
    }

    assert(!thread_.joinable());
    assert(!knock_thread_.joinable());

    state_ = TSStarting;
    thread_ = std::thread([this]() {
      // Block all signals except (maybe) SIG_THREAD_EXIT.
      sigset_t sigset;
      sigfillset(&sigset);
      if (ReliableSignals) {
        // We're a reliable-signal case, so block SIG_THREAD_EXIT
        // too.  The user promises to  be calling pselect/ppoll
        // and unblocking it only for the duration of that call.
        // This means that if it occurs in any other code path
        // in the user's Run(), it will be deferred until the
        // point at which the user calls pselect/ppoll.
      } else {
        // The user promises to call IsExitRequested() after
        // each blocking system call.
        sigdelset(&sigset, ExitSigMask::SIG_THREAD_EXIT);
      }
      ThreadSetSigmask(&sigset);

      // TSReady is really just for debug now; it means we have the
      // signal masks established.
      this->state_ = TSReady;

      // Boilerplate concluded - run the user's code.  Note
      // that it's possible we were told to exit already, though!
      // In particular, without reliable signals, or when Terminate()
      // is called before we get the signals set up, we need this test.
      if (!exit_requested_) {
        this->Run();
      }

      // We're done; remark on this and terminate.
      this->state_ = TSDone;
    });

    return true;
  }

  /*
   * Note that the destructor waits for the thread to
   * finish, if one was started.
   */
  ~SyscallThread() { Terminate(Wait); }

  /*!
   * Do whatever you need done asynchronously here.  Call
   * IsExitRequested() after making blocking system calls
   * (and optonally elsewhere too).
   */
  virtual void Run() = 0;

  /*!
   * If you promised to call pselect/ppoll, use this to get
   * the mask to pass as the sigmask argument.  Note that this
   * function exists only if you promised!
   */
  template <typename T = std::enable_if<ReliableSignals, const sigset_t *>>
  typename T::type Sigmask() const {
    return &InternalMasks().GetExitMask();
  }

  /*
   * If you said "not using reliable signals", use this to
   * defer/disable SIG_THREAD_EXIT for a particular code path
   * that needs to make sure system calls *aren't* interrupted.
   *
   * (If you're using reliable signals, this signal mask is in
   * effect everywhere except where you call pselect/ppoll using
   * Sigmask().)
   */
  template <typename T = std::enable_if<!ReliableSignals, void>>
  typename T::type PushDefer() {
    if (++defer_count_ == 1) {
      ThreadSetSigmask(&InternalMasks().GetAllMask());
    }
  }

  /*
   * If you said "not using reliable signals", use this to
   * re-enable SIG_THREAD_EXIT after pushing a defer.  (If you
   * are going to exit you need not bother.)
   */
  template <typename T = std::enable_if<!ReliableSignals, void>>
  typename T::type PopDefer() {
    if (--defer_count_ == 0) {
      ThreadSetSigmask(&InternalMasks().GetExitMask());
    }
  }

  /*
   * Call this on the thread, from any other thread, to request
   * termination (exit) of the running thread.  Optionally, wait
   * for actual termination.
   *
   * Does nothing if the thread was never started, or is already
   * terminated.  Note, however, that Terminate(true) will
   * wait for the termination to complete, after an earlier
   * Terminate(false).
   */
  void Terminate(enum WaitType waittype = Wait) {
    if (state_ == TSNotStarted) {
      // If never started, there is nothing to terminate.
      return;
    }

    // We should signal the thread if:
    // - it's gotten past TSReady (but this test would race
    //   with the thread itself, so we just assume it has), and
    // - no one else asked it to exit yet, and
    // - it's not already on its way out, or done.
    // (if someone else already asked, that someone-else also
    // kicked off any signalling required).
    bool send_signal = !exit_requested_ && state_ < TSExiting;
    exit_requested_ = true;

    if (send_signal) {
      if (ReliableSignals) {
        KickThread(&thread_);
      } else {
        assert(!knock_thread_.joinable());
        knock_thread_ = std::thread([this]() {
          while (this->state_ < TSExiting) {
            KickThread(&this->thread_);
            // Use nanosleep so that SIG_THREAD_EXIT will interrupt us.
            // The documentation says that sleep_for waits for the full
            // time (presumably using a loop with nanosleep calls).
            //
            // std::this_thread::sleep_for(std::chrono::milliseconds(250));
            struct timespec delay = {.tv_sec = 0, .tv_nsec = 250 * 1000 * 1000};
            nanosleep(&delay, nullptr);
          }
        });
      }
    }

    if (waittype == Wait) {
      WaitFor();
    }
  }

  /*!
   * If you want to verify that the job has finished, call WaitFor.
   * Note, however, that this may result in a scheduling yield (it
   * potentially calls join()).
   *
   * Note that WaitFor() DOES NOT request that the thread terminate;
   * to do that, call Terminate() first.  If you call wait too soon
   * you could wait forever.
   */
  void WaitFor() {
    // If the thread was already joined (redundant Terminate)
    // there's nothing more to do with that part.
    if (thread_.joinable()) {
      thread_.join();
    }
    if (knock_thread_.joinable()) {
      // Kick the knock thread now, to make it finish early.
      KickThread(&knock_thread_);
      knock_thread_.join();
    }
  }

  bool IsExitRequested() { return exit_requested_; }
  void BeginExiting() { state_ = TSExiting; }
  bool Done() { return state_ == TSDone; }

 private:
  // internal function that exists purely to initialize the
  // single global instance of ExitSigMask if and when we need it
  static const ExitSigMask &InternalMasks() {
    static ExitSigMask masks;
    return masks;
  }

  // Kick (deliver signal to) thread so to get any in-progress
  // system call to return EINTR, once the thread takes the signal.
  static inline void KickThread(std::thread *thread) {
    pthread_kill(thread->native_handle(), ExitSigMask::SIG_THREAD_EXIT);
  }

  volatile enum ThreadState state_;
  volatile bool exit_requested_;
  std::thread thread_;
  std::thread knock_thread_;
  int defer_count_;
};

// Give callers nicer names: Pfuncs means pselect/ppoll only
// as the blocking system calls; SyscallThreadAny means we can
// use any system call as a blocking sytem call.
using SyscallThreadPfuncs = SyscallThread<true>;
using SyscallThreadAny = SyscallThread<false>;

}  // namespace bess
}  // namespace utils

#endif  // BESS_UTILS_SYSCALLTHREAD_H
