#pragma once

class Image {

	public:

		Image(int width_arg, int height_arg, int nb_chan_arg);
		Image(const char* path);
		~Image();

		void create_gray_array();

		int get_size();

	private:

		int width;
		int height;
		int nb_chan;

		unsigned char *img_array;
		unsigned char *img_gray_array;

};
