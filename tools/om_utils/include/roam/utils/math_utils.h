#pragma once

// -----------------------------------------------------------------------------------
namespace ROAM
// -----------------------------------------------------------------------------------
{
	class MathUtils
	{
	public:

		/// Compares two double values with desired precision
		/// @param[in] x first variable
		/// @param[in] y second variable
		/// @param[in] ulp desired precision in ULPs (units in the last place)
		/// @return True if x and y are close enough
		// -----------------------------------------------------------------------------------
		template<class T>
		static typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
			almost_equal(const T x, const T y, const int ulp = 2)
		// -----------------------------------------------------------------------------------
		{
			// the machine epsilon has to be scaled to the magnitude of the values used
			// and multiplied by the desired precision in ULPs (units in the last place)
			return std::abs(x - y) < std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp
				// unless the result is subnormal
				|| std::abs(x - y) < std::numeric_limits<T>::min();
		}

		/// Compares two double values with desired precision
		/// @param[in] x first variable
		/// @param[in] y second variable
		/// @param[in] precision rel_eps: abs(a-b)/abs(a) < rel_eps
		/// @param[in] ulp desired precision in ULPs (units in the last place)
		/// @return True if x and y are close enough
		// -----------------------------------------------------------------------------------
		template<class T>
		static typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
			relative_eps_convergence(const T x, const T y, const T rel_eps, const int ulp = 2)
		// -----------------------------------------------------------------------------------
		{
			if(rel_eps < 0.0)
				return false;

			// handle a == b
			if(MathUtils::almost_equal(x, y, ulp))
				return true;

			// othewise let's check the standard rel_eps convergence
			return ((std::abs(x - y) / std::abs(x)) < rel_eps);
		}
	};
}