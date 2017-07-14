#include "roam/utils/timer.h"

using namespace ROAM;

// -----------------------------------------------------------------------------------
Timer::Timer()
// -----------------------------------------------------------------------------------
{
	Start();
	time_span = std::chrono::duration<double>::zero();
}

// -----------------------------------------------------------------------------------
void Timer::Start()
// -----------------------------------------------------------------------------------
{
	t_start = std::chrono::high_resolution_clock::now();
}

// -----------------------------------------------------------------------------------
double Timer::Stop()
// -----------------------------------------------------------------------------------
{
	const std::chrono::high_resolution_clock::time_point t_end = std::chrono::high_resolution_clock::now();
	time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t_end - t_start);
	return time_span.count();
}

// -----------------------------------------------------------------------------------
double Timer::LastTimeSpan() const
// -----------------------------------------------------------------------------------
{
	return time_span.count();
}
