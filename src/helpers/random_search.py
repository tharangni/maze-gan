import os
import csv
import glob
import random
import importlib
import helpers.evaluation as ev
from time import gmtime, strftime

csv_file = None
csv_writer = None

def begin_search(opt: {}):
    global csv_writer
    global csv_file

    iterations = 200
    batch_range = [0, 200]
    d_lr_range = [0.003, 2]
    g_lr_range = [0.003, 2]
    #img_size_range = [28, 64]
    maze_size_range = [2, 8]
    latent_dim_range = [2, 200]
    temp_range = [0.01, 10]

    path = os.path.join('.', 'models', opt.model, 'random_search_results')
    os.makedirs(path, exist_ok=True)

    csv_file = open(os.path.join(path,  strftime("%Y-%m-%d %H-%M-%S", gmtime()), ".csv"), 'w+', newline='')
    csv_writer = csv.writer(csv_file, delimiter=',')  # for looging results for graphing.
    csv_writer.writerow(['model', 'batch_size', 'd_lr', 'g_lr', 'latent_size', 'temp_size', 'epoch_no', 'd_loss', 'g_loss', 'D(x)', 'D(G(X))', 'correct_amount'])

    for i in range(0, iterations):
        batch_size = random.randint(batch_range[0], batch_range[1])
        d_lr_size = random.randint(d_lr_range[0], d_lr_range[1])
        g_lr_size = random.randint(g_lr_range[0], g_lr_range[1])
        maze_size = random.randint(maze_size_range[0], maze_size_range[1])
        latent_size = random.randint(latent_dim_range[0], latent_dim_range[1])
        temp_size = random.randint(temp_range[0], temp_range[1])

        opt.batch_size = batch_size
        opt.d_lr = d_lr_size
        opt.g_lr = g_lr_size
        opt.latent_dim = latent_size
        opt.temp = temp_size

        module_path = os.path.abspath(os.path.join('models', opt.dataset, opt.model))
        samples_path = os.path.abspath(
            os.path.join('models', opt.dataset, opt.model, 'samples', run, '*.sample.tar'))
        sample_files = glob.glob(samples_path)
        sample_files.sort()

        print("Iteration :", i, "/", iterations,"    opt: ", opt)
        model = importlib.import_module('.'.join(['models', opt.model, opt.model]))
        model.run(opt)

        #get sample session
        #correct_amount = ev.check_ind(sample_files, run)
        #log correct amount save_results(model.LOGGER, opt, correct_amount)

def save_results(model, opt, correct_amount):
    global csv_writer
    GAN_stats = model.LOGGER.latest_GAN_stats

    csv_writer.writerow(
         opt.model,
         opt.batch_size,
         opt.d_lr,
         opt.g_lr,
         opt.latent_size,
         opt.temp_size,
         GAN_stats['epoch_no'],
         GAN_stats['d_loss'],
         GAN_stats['g_loss'],
         GAN_stats['D(x)'],
         GAN_stats['D(G(X))'],
         correct_amount)
    # no of corrected
    model.LOGGER.last_GAN_stats