import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from data.bouncing_ball.bouncing_ball import bounce_vec
from RNN.hf_rnn import RNNHfOptim


def gen_dataset(video_res, t_steps, n_samples=100):
    print "generating dataset..."
    radius = 2
    n_balls = 2
    seq = np.array([bounce_vec(video_res, n=n_balls, r=[radius] * n_balls, T=t_steps) for i in range(n_samples)], dtype='float32')
    target = np.array(seq[:, 1:, :], dtype='float32')
    seq = seq[:, :-1, :]

    return (seq, target)


def test(model, video_res, t_steps, n_samples):
    seq_testing, target_testing = gen_dataset(video_res, t_steps)
    guess = np.array([model.predict(s) for s in seq_testing], dtype='float32')
    print guess.shape

    rms = sqrt(mean_squared_error(guess, target_testing))
    print "erreur ", rms


def main():
    video_res = 15
    t_steps = 30
    n_in = n_out = video_res ** 2
    n_samples = 1000

    #generating training_set
    seq_training, target_training = gen_dataset(video_res, t_steps, n_samples)

    print "preparing optimizer ..."
    trainer = RNNHfOptim(n_in=n_in, n_out=n_out, n_hidden=300)
    trainer.tune_optimizer(num_updates=19)

    print "training..."
    trainer.fit(seq_training, target_training)

    #evaluating results
    print "testing..."
    test(trainer, video_res, t_steps, n_samples)


if __name__ == "__main__":
    main()


