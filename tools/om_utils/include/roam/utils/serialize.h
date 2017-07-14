#pragma once

#include <iostream>
#include <fstream>
#include <yaml-cpp/yaml.h>

// -----------------------------------------------------------------------------------
namespace ROAM
// -----------------------------------------------------------------------------------
{
	/// \brief Virtual class, serializer of Parameter structs
	// -----------------------------------------------------------------------------------
	class Serialize
	// -----------------------------------------------------------------------------------
	{
	public:
		// -----------------------------------------------------------------------------------
		virtual ~Serialize()
		// -----------------------------------------------------------------------------------
		{
		}

		/// Loads a given node from config file
		/// @param[in] filename Full path to config file
		/// @param[in] key Map key
		// -----------------------------------------------------------------------------------
		virtual void Load(const std::string &filename, const std::string &key)
		// -----------------------------------------------------------------------------------
		{
			YAML::Node config = YAML::LoadFile(filename);
			if(!config)
				throw std::invalid_argument("GreedyKSDPP::Parameters::Load - cannot open file: " + filename);

			doLoad(config, key);
		}
		
		/// Saves a given node to config file
		/// @param[in] filename Full path to config file
		/// @param[in] append (optional) If true, only appends, otherwise creates a new file
		// -----------------------------------------------------------------------------------
		virtual void Save(const std::string &filename, const bool append = false) const
		// -----------------------------------------------------------------------------------
		{
			const std::ios_base::openmode mode = append ? (std::ios_base::out | std::ios_base::app) : std::ios_base::out;
			std::ofstream fout(filename, mode);

			if(!fout.is_open())
				std::cout << "Parameters::Save - can't open file: " + filename;

			Print(fout);
		}

		/// Prints parameters
		/// @param[in] out (optional) Output stream (otherwise console)
		// -----------------------------------------------------------------------------------
		virtual void Print(std::ostream &out = std::cout) const = 0;
		// -----------------------------------------------------------------------------------

		/// Prints parameters into stream
		// -----------------------------------------------------------------------------------
		friend std::ostream& operator<<(std::ostream& stream, const Serialize& params)
		// -----------------------------------------------------------------------------------
		{
			params.Print(stream); 
			return stream;
		}

	protected:
		/// Load parameters
		/// @param[in] config Config file
		/// @param[in] key Map key
		// -----------------------------------------------------------------------------------
		virtual void doLoad(YAML::Node &config, const std::string &key) = 0;
		// -----------------------------------------------------------------------------------

		/// Loads respective key
		/// @param[in] node Requested node
		/// @param[in] key Requested key
		/// @param[out] value Value
		/// @retval TRUE If key was found
		/// @retval FALSE Otherwise
		// -----------------------------------------------------------------------------------
		template <typename IN, typename OUT>
		bool readKey(const YAML::Node &node, const std::string &key, OUT &value) const
		// -----------------------------------------------------------------------------------
		{
			if(node[key])
			{
				value = static_cast<OUT>(node[key].as<IN>());
				return true;
			}

			return false;
		}

		/// Writes respective key
		/// @param emitter Yaml stream
		/// @param[in] key Key
		/// @param[in] value Value
		/// @param[in] comment Optional comment
		// -----------------------------------------------------------------------------------
		template <typename OUT>
		void writeKey(YAML::Emitter &emitter, const std::string &key, const OUT &value,
					  const std::string &comment = std::string()) const
		// -----------------------------------------------------------------------------------
		{
			emitter << YAML::Key << key << YAML::Value << value << YAML::Comment(comment);
		}

		/// Prints section separator
		/// @param emitter Yaml stream
		/// @param[in] name Text to be printed
		// -----------------------------------------------------------------------------------
		void printSectionHeader(YAML::Emitter &emitter, const std::string &name) const
		// -----------------------------------------------------------------------------------
		{
			emitter << YAML::Comment("---------------------------------------------------------------------");
			emitter << YAML::Newline;
			emitter << YAML::Comment(name);
			emitter << YAML::Newline;
			emitter << YAML::Comment("---------------------------------------------------------------------");
			emitter << YAML::Newline;
		}
	};
}