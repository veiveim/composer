#include <iostream>
#include <fstream>
#include <dirent.h>
#include <string.h>

#include "wave_explorer.hpp"

const int kwaves = 256;
const double PI = 3.141592653589793238460;

int main(int argc, char *argv[])
{
	if (argc != 3) {
		std::cout << "usage: a.out song_file_dir wave_file_dir\n";
		return 1;
	}

	WaveExplorer Waver;
	std::ofstream outfile;

	DIR *root_dir = opendir(argv[1]);
	if (root_dir == NULL) {
		std::cout << "Error: opendir failed\n";
		return 1;
	}

	struct dirent *root_ent;
	while ((root_ent = readdir(root_dir)) != NULL) {
		char *singer = root_ent->d_name;
		if (!strcmp(singer, ".") || !strcmp(singer, ".."))
			continue;
		
		char singer_path[128];
		sprintf(singer_path, "%s/%s", argv[1], singer);
		DIR *sub_dir = opendir(singer_path);
		if (sub_dir == NULL) {
			std::cout << "Error: open sub_dir failed\n";
			continue;
		}

		struct dirent *sub_ent;
		while ((sub_ent = readdir(sub_dir)) != NULL) {
			char *song = sub_ent->d_name;
			if (!strcmp(song, ".") || !strcmp(song, ".."))
				continue;

			char song_path[128];
			char wave_path[128];
			sprintf(song_path, "%s/%s", singer_path, song);
			sprintf(wave_path, "%s/%s.txt", argv[2], strtok(song, "."));
			std::cout << singer_path << ' ' << song_path << ' ' << wave_path << std::endl;

			int duration_sec = Waver.load_file(song_path) / AV_TIME_BASE;
			outfile.open(wave_path);

			for (int i = 0; i < duration_sec * 40; i++){
				const WaveSequence *waves = Waver.get_next_frame_waves();

				if (waves == NULL)
					continue;

				int k = 0;
				for (int j = 0; k < kwaves; j++) {
					// FFT waves is coupled, waves (j, N - j) have same big amplitude, we count j.
					if (waves->index[j] > (NB_SAMPLES_PER_FRAME / 2)) {
						continue;
					}
					//outfile << int(waves->real[j] / 100) << ',' <<  int(waves->imag[j] / 100) << ',' << waves->index[j] << '\t';
					
					float scale = sqrt((pow(waves->real[j], 2) + pow(waves->imag[j], 2))) / NB_SAMPLES_PER_FRAME;
        			float theta = atan2(waves->imag[j], waves->real[j]);
        			float cycle = 2.0 * PI * waves->index[j] / NB_SAMPLES_PER_FRAME;
					outfile << scale / 1000 << ',' <<  theta << ',' << cycle * 10 << '\t';
					k++;
				}
				outfile << '\n';

				if (i != 0 && (i % 100 == 0)) {
					char pict_path[128];
					sprintf(pict_path, "pictures/frame_%d_rii/%s_%d", NB_SAMPLES_PER_FRAME, song, i);
					std::cout << "Draw: " << pict_path << std::endl;
					// draw coupled waves, so * 2
					Waver.draw_frames(pict_path, kwaves * 2);
				}
			}	

			outfile.close();
			Waver.close_file();
		}	
	}
	return 0;
}