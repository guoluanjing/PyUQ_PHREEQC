import settings
import batch_simulations
import processing

import time

start_time = time.time()

settings.init()

# batch_simulations.run_simulations()

processing.process()

print("Total run time = %.2f seconds." % (time.time() - start_time))
