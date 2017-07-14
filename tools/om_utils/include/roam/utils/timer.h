#pragma once
#include <chrono>

// -----------------------------------------------------------------------------------
namespace ROAM
// -----------------------------------------------------------------------------------
{
	/// \brief Utility class providing high resolution timer
	///
	///  Provides a simple wrapper around std::chrono
	// -----------------------------------------------------------------------------------
	class Timer
	// -----------------------------------------------------------------------------------
	{
	public:
		/// Defautl constructor starts timer
		Timer();

		/// Starts the clock
		void Start();

		/// Stops the clock
		/// @return Returns the timespan in seconds
		double Stop();

		/// @return Returns the last timespan in seconds
		double LastTimeSpan() const;
		
	protected:
		std::chrono::high_resolution_clock::time_point t_start;
		std::chrono::duration<double> time_span;
	};
}