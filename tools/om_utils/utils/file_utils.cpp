#include "roam/utils/file_utils.h"

using namespace ROAM;

// -----------------------------------------------------------------------------------
bool FileUtils::FileExists(const std::string &filename)
// -----------------------------------------------------------------------------------
{
    struct stat buf;
    if (stat(filename.c_str(), &buf) != -1)
    {
        return true;
    }
    return false;
}

// -----------------------------------------------------------------------------------
bool FileUtils::MkDir(const std::string &dir)
// -----------------------------------------------------------------------------------
{
#if defined _MSC_VER
	if(_mkdir(dir.data()) == 0)
		return true;
#elif defined __GNUC__
	if(mkdir(dir.data(), 0777) == 0)
		return true;
#endif
	return false;
}

// -----------------------------------------------------------------------------------
std::string FileUtils::GetFullName(const std::string &dir, const std::string &filename)
// -----------------------------------------------------------------------------------
{		
    return dir + "/" + filename;
}

// -----------------------------------------------------------------------------------
std::string FileUtils::GetFullName(const std::string &dir, const std::string &basename, 
											  const std::string &extensFileUtilsn)
// -----------------------------------------------------------------------------------
{		
    return dir + "/" + basename + "." + extensFileUtilsn;
}

// -----------------------------------------------------------------------------------
std::vector<std::string> FileUtils::ReadBasenames(const std::string &filename)
// -----------------------------------------------------------------------------------
{
	std::vector<std::string> basenames;
	ReadBasenames(filename, basenames);
	return basenames;
}

// -----------------------------------------------------------------------------------
void FileUtils::ReadBasenames(const std::string &filename, std::vector<std::string> &basenames)
// -----------------------------------------------------------------------------------
{
    std::ifstream file(filename.c_str());
	if(!file.is_open())
		throw std::invalid_argument(std::string("readBasenames: can't open file: " + filename));
		    
	while(file.good())
	{
		std::string line;
		std::getline (file, line);
		if(!line.compare(""))
			continue;

		const std::string basename = RemoveExtension(line);
		basenames.push_back(basename);
	}
	file.close();
}

// -----------------------------------------------------------------------------------
std::string FileUtils::RemoveExtension(const std::string &filename)
// -----------------------------------------------------------------------------------
{
    std::string::const_reverse_iterator pivot = std::find(filename.rbegin(), filename.rend(), '.');
    return pivot == filename.rend() ? filename : std::string(filename.begin(), pivot.base() - 1);
}

// -----------------------------------------------------------------------------------
void FileUtils::SaveVectorBinary(const std::vector<double> &vct, const std::string &filename)
// -----------------------------------------------------------------------------------
{
	const size_t n_dims = vct.size();

	FILE *f = fopen(filename.c_str(), "wb");
	if(f == nullptr)
		throw std::invalid_argument("FileUtils::saveVectorBinary: cannot open file:" + filename);

	fwrite(&n_dims, sizeof(int), 1, f);
	fwrite(vct.data() , sizeof(double), n_dims, f);

	fclose(f);
}

// -----------------------------------------------------------------------------------
void FileUtils::LoadVectorBinary(std::vector<double> &vector, const std::string &filename)
// -----------------------------------------------------------------------------------
{
	if(!FileExists(filename)) 
		throw std::invalid_argument(std::string("FileUtils::loadVectorBinary: file doesn't exist:" + filename));

	FILE *f = fopen(filename.c_str(), "rb");
	if(f == nullptr)
		throw std::invalid_argument("FileUtils::loadVectorBinary: cannot open file:" + filename);

	int n_dims = 0;
    fread(&n_dims, 1, sizeof(int), f);

	vector.resize(n_dims);
	fread(vector.data(), sizeof(double), n_dims, f);

	fclose(f);
}

// -----------------------------------------------------------------------------------
void FileUtils::SaveVectorASCII(const std::vector<double> &vct, const std::string &filename)
// -----------------------------------------------------------------------------------
{
	const size_t n_dims = vct.size();

	FILE *f = fopen(filename.c_str(), "w");
	if(f == nullptr)
		throw std::invalid_argument("FileUtils::saveVectorASCII: cannot open file:" + filename);

    fprintf(f, "%d\n", static_cast<int>(vct.size()));

	for(size_t i=0; i < vct.size(); ++i)
	{
		fprintf(f, "%0.8f", vct[i]);
		fprintf(f, " ");
	}

	fclose(f);
}
