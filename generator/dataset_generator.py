import generator_lib
import time
from datetime import timedelta

start = time.time()

# CONFIGURE THIS
MODE = "default"
DATABASE_PATH = "../physionet.org/files/ucddb/1.0.0/"
STORAGE_PATH = "C:/Users/saman/Documents/storage/"
FRAME_LENGTH = 8
FRAME_SHIFT = 5
IMAGING_SOLUTION = "RP"
THRESHOLD = 0.12
SAMPLE_FREQUENCY = 128


generator_lib.generate(
    MODE,
    DATABASE_PATH,
    STORAGE_PATH,
    SAMPLE_FREQUENCY,
    FRAME_LENGTH,
    FRAME_SHIFT,
    IMAGING_SOLUTION,
    THRESHOLD
)

end = time.time()
print(f"Elapsed time: {timedelta(seconds=end-start)}")
