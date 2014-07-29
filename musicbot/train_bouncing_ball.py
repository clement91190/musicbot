import numpy as np
import cPickle
from math import sqrt
from sklearn.metrics import mean_squared_error
from data.bouncing_ball.bouncing_ball import bounce_vec, kl_seq
from RNN.hf_rnn import RNNHfOptim
from RNN.SGRNN import WeightsHandler


def gen_dataset(video_res, t_steps, n_samples=100):
    print "generating dataset..."
    radius = 2
    n_balls = 3
    seq = np.array([bounce_vec(video_res, n=n_balls, r=[radius] * n_balls, T=t_steps) for i in range(n_samples)], dtype='float32')
    target = np.array(seq[:, 1:, :], dtype='float32')
    seq = seq[:, :-1, :]

    return (seq, target)


def test(model, video_res, t_steps, n_samples):
    seq_testing, target_testing = gen_dataset(video_res, t_steps)
    guess = np.array([model.predict(s) for s in seq_testing], dtype='float32')
    print guess.shape

    rms = sqrt(mean_squared_error(guess, target_testing))
    print "erreur rmse :", rms

    kl_err = np.mean([kl_seq(Vp, Vq) for (Vp, Vq) in zip(guess, target_testing)])
    print "erreur kl-div : ", kl_err
    return kl_err


def train_classical_rnn():
    video_res = 15
    t_steps = 30
    n_in = n_out = video_res ** 2
    n_samples = 100

    #generating training_set
    seq_training, target_training = gen_dataset(video_res, t_steps, n_samples)
    seq_training = seq_training * 2
    target_training = target_training * 2
    n_updates = 100

    print "preparing optimizer ..."
    trainer = RNNHfOptim(n_in=n_in, n_out=n_out, n_hidden=300, model="RNN", activation='sigmoid')
    trainer.tune_optimizer(
        num_updates=n_updates, cg_number_batches=1, gd_number_batches=1,
        save_progress="temp_saving_file.pkl", plot_cost_file="plot_cost_value.pkl")

    test_errors = []

    print "training..."
    trainer.prepare(seq_training, target_training)
    for i in range(n_updates):
        trainer.train_step(i)
        test_errors.append(test(trainer, video_res, t_steps, 50))
        with open('test_errors.pkl', 'w') as f:
            cPickle.dump(test_errors, f)

    #evaluating results
    print "testing..."
    test(trainer, video_res, t_steps, n_samples)


def main():
    video_res = 18
    t_steps = 30
    n_in = n_out = video_res ** 2
    n_samples = 100
    n_updates = 20

    test_errors = []

    #generating training_set
    seq_training, target_training = gen_dataset(video_res, t_steps, n_samples)

    print "preparing optimizer ..."
    #trainer = RNNHfOptim(n_in=n_in, n_out=n_out, n_hidden=300, model="RNN")
    weight_handler = WeightsHandler(n_in=n_in, n_out=n_out, n_hidden_start=300)
    trainer = RNNHfOptim(model="SGRNN", weight_handler=weight_handler, activation='sigmoid')
    trainer.tune_optimizer(
        num_updates=n_updates, cg_number_batches=5, gd_number_batches=5,
        save_progress="temp_saving_file.pkl", plot_cost_file="plot_cost_value.pkl")

    print "training..."
    trainer.prepare(seq_training, target_training)
    for i in range(n_updates):
        trainer.train_step(i)
        test_errors.append(test(trainer, video_res, t_steps, 50))
        with open('test_errors.pkl', 'w') as f:
            cPickle.dump(test_errors, f)

    #evaluating results
    print "testing..."
    test(trainer, video_res, t_steps, n_samples)


if __name__ == "__main__":
    #main()
    train_classical_rnn()


