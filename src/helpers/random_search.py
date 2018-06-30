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
    img_size_range = [28, 64]
    maze_size_range = [2, 8]
    latent_dim_range = [2, 200]
    temp_range = [0.01, 10]

    path = os.path.join('.', 'models', opt.model, 'random_search_results')
    os.makedirs(path, exist_ok=True)
    file_name = strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + ".csv"
    print(file_name)
    csv_file = open(os.path.join(path, file_name), 'w+', newline='')
    csv_writer = csv.writer(csv_file, delimiter=',')  # for looging results for graphing.
    csv_writer.writerow(
        ['model', 'batch_size', 'd_lr', 'g_lr', 'latent_size', 'temp_size', 'epoch_no', 'd_loss', 'g_loss', 'D(x)',
         'D(G(X))', 'correct_amount'])
    # TODO add batch headings for correct results

    for i in range(0, iterations):
        batch_size = random.randint(batch_range[0], batch_range[1])
        d_lr_size = random.uniform(d_lr_range[0], d_lr_range[1])
        g_lr_size = random.uniform(g_lr_range[0], g_lr_range[1])
        maze_size = random.randint(maze_size_range[0], maze_size_range[1])
        latent_size = random.randint(latent_dim_range[0], latent_dim_range[1])
        temp_size = random.uniform(temp_range[0], temp_range[1])

        #opt.batch_size = batch_size
        opt.d_lr = d_lr_size
        opt.g_lr = g_lr_size
        opt.latent_dim = latent_size
        opt.temp = temp_size

        print("Iteration :", i, "/", iterations, "    opt: ", opt)

        # get sample session
        model = importlib.import_module('.'.join(['models', opt.model, opt.model]))
        model.run(opt)

        samples_path = os.path.abspath(
            os.path.join('models', opt.model, 'samples', model.LOGGER.run, '*.sample.tar'))
        sample_files = glob.glob(samples_path)
        sample_files.sort()

        correct_amount = ev.check_ind(sample_files)
        save_results(model.LOGGER, opt, correct_amount)
    close_file()

def save_results(logger, opt, correct_amount):
    global csv_writer
    GAN_stats = logger.lastest_GAN_stats

    row = [opt.model,
        opt.batch_size,
        opt.d_lr,
        opt.g_lr,
        opt.latent_dim,
        opt.temp,
        GAN_stats['epoch'],
        GAN_stats['d_loss'],
        GAN_stats['g_loss'],
        GAN_stats['d_x'],
        GAN_stats['d_g_z']]
    row.extend(correct_amount)
    csv_writer.writerow(row)
    # no of corrected

def close_file():
    csv_file.close()