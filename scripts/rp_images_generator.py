import generator_lib

# CONFIGURE THIS
MODE = "default"
DATABASE_PATH = "E:\\tcc\\base1\\physionet.org\\files\\ucddb\\1.0.0\\"
STORAGE_PATH = "E:\\tcc\\storage"
FRAME_LENGTH = 5
FRAME_SHIFT = 2
THRESHOLD = 0.12
SAMPLE_FREQUENCY = 128

generator = generator_lib.RPImageGenerator(MODE,
                                           DATABASE_PATH,
                                           STORAGE_PATH,
                                           FRAME_LENGTH,
                                           FRAME_SHIFT,
                                           THRESHOLD,
                                           SAMPLE_FREQUENCY)

generator.generate()
