#pragma once
#include <vector>
#include <string>
#include <fstream>

#if defined _MSC_VER
#include <direct.h>
#endif

#include <sys/types.h>
#include <sys/stat.h>

#include <iostream>
#include <stdexcept>
#include <algorithm>

// -----------------------------------------------------------------------------------
namespace ROAM
// -----------------------------------------------------------------------------------
{
	/// \brief Utility class providing basic input/output helper files
	///
	///  Provides routines for fullfile paths, checking paths, reading basenames and saving/reading binary files.
	// -----------------------------------------------------------------------------------
	class FileUtils
	// -----------------------------------------------------------------------------------
	{
	public:

		/// Checks existence of a path 
		/// @param[in] filename full path to a given file
		/// @return True if the file exists
		static bool FileExists(const std::string &filename);

		/// Creates new directory
		/// @param[in] dir full path to requested directory
		/// @return true on success
		static bool MkDir(const std::string &dir);

		/// Creates a full path 
		/// @param[in] dir directory (without the last "/")
		/// @param[in] filename (including extention)
		/// @return Full path ("dir/filename")
		static std::string GetFullName(const std::string &dir, const std::string &filename);

		/// Creates a full path 
		/// @param[in] dir directory (without the last "/")
		/// @param[in] basename (without extention)
		/// @param[in] extension (without ".")
		/// @return Full path ("dir/filename.extension")
		static std::string GetFullName(const std::string &dir, const std::string &basename, const std::string &extension);
	 
		/// Reads basenames from a file
		/// @param[in] filename Fullpath to a file
		/// @return Vector of basenames
		static std::vector<std::string> ReadBasenames(const std::string &filename);

		/// Reads basenames from a file
		/// @param[in] filename Fullpath to a file
		/// @param[out] basenames Vector of basenames
		static void ReadBasenames(const std::string &filename, std::vector<std::string> &basenames); 

		/// Extracts a basename
		/// @param[in] filename
		/// @return Basename (i.e. removes the extension)
		static std::string RemoveExtension(const std::string &filename);

		/// Saves vector in ascii format
		/// @param[in] vector Vector of values of type double
		/// @param[in] filename Fullpath to a file
		static void SaveVectorASCII(const std::vector<double> &vector, const std::string &filename);

		/// Saves vector in binary format
		/// @param[in] vector Vector of values of type double
		/// @param[in] filename Fullpath to a file
		static void SaveVectorBinary(const std::vector<double> &vector, const std::string &filename);

		/// Load vector from binary format
		/// @param[in] filename Fullpath to a file
		/// @param[out] vector Vector of values of type double
		static void LoadVectorBinary(std::vector<double> &vector, const std::string &filename);
	};
}


