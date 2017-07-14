#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <map>
#include <set>
#include <algorithm>

#include "file_utils.h"

// -----------------------------------------------------------------------------------
namespace ROAM
// -----------------------------------------------------------------------------------
{
	/// \brief Class providing basic label <-> color (uchar) conversions
	// -----------------------------------------------------------------------------------
	class ColorMap
	// -----------------------------------------------------------------------------------
	{
	public:
		/// Default constructor (optionally loads the colormap)
		/// @param[in] filename full path to a colormap file
		// -----------------------------------------------------------------------------------
		explicit ColorMap(const std::string &filename = std::string()) : m_initialized(false)
		// -----------------------------------------------------------------------------------
		{
			if(!filename.empty())
				Load(filename);
		}

		/// Loads the colormap
		/// @param[in] filename full path to a colormap file
		// -----------------------------------------------------------------------------------
		void Load(const std::string &filename)
		// -----------------------------------------------------------------------------------
		{
			m_label2name.clear();
			m_label2rgb.clear();
			m_name2label.clear();
			m_rgb2label.clear();

			if(!FileUtils::FileExists(filename))
				throw std::invalid_argument("ColorMap::load -  file doesn't exist: " + filename);

			std::ifstream input(filename.c_str());
			if(!input.is_open())
				throw std::invalid_argument("ColorMap::load - could not open file: " + filename);

			while(input.good())
			{
				int cmd = input.peek();
				if(cmd == '#')
				{
					char line[256];
					input.getline(line, 256);
					continue;	// line with comment
				}

				int label = -1;
				std::string name;
				std::vector<int> rgb(3, 0);

				input >> label >> name >> rgb[0] >> rgb[1] >> rgb[2];

				if(!name.compare(""))
					continue;

				// to lower
				std::transform(name.begin(), name.end(), name.begin(), ::tolower);

				m_label2name[label] = name;
				m_label2rgb[label] = rgb;
				m_name2label[name] = label;
				m_rgb2label[rgb] = label;

				if(input.eof())
					break;
			}
			input.close();

			m_labels = m_label2name.size();

			m_initialized = true;
		}

		/// Converts RGB color-code to label
		/// @param[in] rgb color (uchar)
		/// @return class label
		// -----------------------------------------------------------------------------------
		template <typename T>
		int RGB2Label(const std::vector<T> &rgb) const
		// -----------------------------------------------------------------------------------
		{
			if(rgb.size() != 3)
				throw std::invalid_argument("ColorMap::RGB2Label - pixel does not represent RGB, #dims: " + rgb.size());

			typename std::map<std::vector<T>, int>::const_iterator iter = m_rgb2label.find(rgb);
			if(iter == m_rgb2label.end())
			{
				std::string err = "ColorMap::RGB2Label - unknown RGB: [" + std::to_string(rgb[0]) + " "
					+ std::to_string(rgb[1]) + " " + std::to_string(rgb[2]) + "]";
				//throw std::invalid_argument(err.c_str());
				return -1;
			}

			return iter->second;
		}

		/// Converts class label to RGB-code (uchar)
		/// @param[in] label class label
		/// @return RGB color (uchar)
		// -----------------------------------------------------------------------------------
		std::vector<int> Label2RGB(const int label) const
		// -----------------------------------------------------------------------------------
		{
			std::map<int, std::vector<int> >::const_iterator iter = m_label2rgb.find(label);
			if(iter == m_label2rgb.end())
				throw std::invalid_argument("ColorMap::Label2RGB - unknown label");

			return iter->second;
		}

		/// Converts class label to class name
		/// @param[in] label class label
		/// @return class name
		// -----------------------------------------------------------------------------------
		const std::string& Label2Name(const int label) const
		// -----------------------------------------------------------------------------------
		{
			std::map<int, std::string>::const_iterator iter = m_label2name.find(label);
			if(iter == m_label2name.end())
				throw std::invalid_argument("ColorMap::Label2Name - unknown label");

			return iter->second;
		}

		/// Converts class name to class label
		/// @param[in] name class name
		/// @return class label
		// -----------------------------------------------------------------------------------
		int Name2Label(const std::string name) const
		// -----------------------------------------------------------------------------------
		{
			// to lower
			std::string l_name = name;
			std::transform(l_name.begin(), l_name.end(), l_name.begin(), ::tolower);

			std::map<std::string, int>::const_iterator iter = m_name2label.find(l_name);
			if(iter == m_name2label.end())
				throw std::invalid_argument("ColorMap::Label2Name - unknown label");

			return iter->second;
		}

		/// Converts Ground-truth image to class labels
		/// @param[in] rgb GT image
		/// @return Vector of class labels
		// -----------------------------------------------------------------------------------
		template <typename T>
		std::vector<T> GT2Labels(const std::vector<unsigned char> &rgb) const
		// -----------------------------------------------------------------------------------
		{
			std::vector<T> labels;
			GT2Labels(rgb, labels);
			return labels;
		}

		/// Converts Ground-truth image to class labels
		/// @param[in] rgb GT image
		/// @param[out] labels Vector of class labels
		// -----------------------------------------------------------------------------------
		template <typename T>
		void GT2Labels(const std::vector<unsigned char> &rgb, std::vector<T> &labels) const
		// -----------------------------------------------------------------------------------
		{
			const size_t n_pixels = rgb.size() / 3;
			labels.resize(n_pixels, 0);

			#pragma omp parallel for
			for(auto i = 0; i < n_pixels; ++i)
			{
				std::vector<int> pixel_rgb(3, 0);
				pixel_rgb[0] = static_cast<int>(rgb[i * 3 + 0]);
				pixel_rgb[1] = static_cast<int>(rgb[i * 3 + 1]);
				pixel_rgb[2] = static_cast<int>(rgb[i * 3 + 2]);

				labels[i] = static_cast<T>(RGB2Label(pixel_rgb));
			}
		}
		
		/// Converts class labels to color-coded labels
		/// @param[in] labels Vector of class labels
		/// @return Vector of RGB-coded labels
		// -----------------------------------------------------------------------------------
		template <typename T, typename K>
		std::vector<K> Labels2RGB(const std::vector<T> &labels) const
		// -----------------------------------------------------------------------------------
		{
			std::vector<K> rgb;
			Labels2RGB(labels, rgb);
			return rgb;
		}

		/// Converts class labels to color-coded labels
		/// @param[in] labels Vector of class labels
		/// @param[out] rgb Vector of RGB-coded labels
		// -----------------------------------------------------------------------------------
		template <typename T, typename K>
		void Labels2RGB(const std::vector<T> &labels, std::vector<K> &rgb) const
		// -----------------------------------------------------------------------------------
		{
			const size_t n_pixels = labels.size();

			if(n_pixels != (rgb.size() / 3))
				rgb.resize(n_pixels * 3);

			#pragma omp parallel for
			for(auto pixel = 0; pixel < n_pixels; ++pixel)
			{
				const std::vector<int> pixel_rgb = Label2RGB(labels[pixel]);
				rgb[pixel * 3 + 0] = static_cast<K>(pixel_rgb[0]);
				rgb[pixel * 3 + 1] = static_cast<K>(pixel_rgb[1]);
				rgb[pixel * 3 + 2] = static_cast<K>(pixel_rgb[2]);
			}
		}

		/// Overlays RGB image with RGB-coded class labels
		/// @param[in] labels Vector of class labels
		/// @param[in] image RGB image
		/// @param[in] mix Mixing coefficient: mix * class + (1 - mix) * image
		/// @param[in] ignoreLabels List of class names that should be omitted
		/// @return Image overlayed with class labels
		// -----------------------------------------------------------------------------------
		template <typename T>
		std::vector<unsigned char> Labels2Overlay(const std::vector<T> &labels, const unsigned char *image,
												  const double mix = 0.4, 
												  const std::vector<std::string> &ignoreLabels = std::vector<std::string>()) const
		// -----------------------------------------------------------------------------------
		{
			std::vector<unsigned char> rgb;
			Labels2Overlay(labels, image, rgb, mix, ignoreLabels);
			return rgb;
		}

		/// Overlays RGB image with RGB-coded class labels
		/// @param[in] labels Vector of class labels
		/// @param[in] image RGB image
		/// @param[out] rgb Image overlayed with class labels
		/// @param[in] mix Mixing coefficient: mix * class + (1 - mix) * image
		/// @param[in] ignoreLabels List of class names that should be omitted
		// -----------------------------------------------------------------------------------
		template <typename T>
		void Labels2Overlay(const std::vector<T> &labels, const unsigned char *image,
							std::vector<unsigned char> &rgb, const double mix = 0.4,
							const std::vector<std::string> &ignoreLabels = std::vector<std::string>()) const
		// -----------------------------------------------------------------------------------
		{
			const size_t n_pixels = labels.size();

			if(n_pixels != (rgb.size() / 3))// || n_pixels != (image.size() / 3))
				rgb.resize(3 * n_pixels);
			//throw std::invalid_argument("ColorMap::Labels2Overlay dimensions must agree");

			// mix <0, 1.0>
			const double trunc_mix = std::max(std::min(mix, 1.0), 0.0);

			std::set<int> ignore;
			for(size_t i = 0; i < ignoreLabels.size(); ++i)
				ignore.insert(Name2Label(ignoreLabels[i]));

			for(size_t pixel = 0; pixel < n_pixels; ++pixel)
			{
				double alpha = trunc_mix;

				// ignore label?
				std::set<int>::const_iterator iter = ignore.find(labels[pixel]);
				if(iter != ignore.end())
					alpha = 0.0;

				const std::vector<int> pixel_rgb = Label2RGB(labels[pixel]);
				rgb[pixel * 3 + 0] = static_cast<unsigned char>(alpha * pixel_rgb[0] + (1.0 - alpha) * image[pixel * 3 + 0]);
				rgb[pixel * 3 + 1] = static_cast<unsigned char>(alpha * pixel_rgb[1] + (1.0 - alpha) * image[pixel * 3 + 1]);
				rgb[pixel * 3 + 2] = static_cast<unsigned char>(alpha * pixel_rgb[2] + (1.0 - alpha) * image[pixel * 3 + 2]);
			}
		}

		/// @return Returns the number of classes
		// -----------------------------------------------------------------------------------
		size_t NClasses() const
		// -----------------------------------------------------------------------------------
		{
			return m_labels;
		}

		/// @return Returns true if the colormap was loaded
		// -----------------------------------------------------------------------------------
		bool IsInitialized() const
		// -----------------------------------------------------------------------------------
		{
			return m_initialized;
		}

	protected:
		std::map<int, std::string> m_label2name;
		std::map<int, std::vector<int> > m_label2rgb;

		std::map<std::string, int> m_name2label;
		std::map<std::vector<int>, int> m_rgb2label;
		size_t m_labels;

		bool m_initialized;
	};
}