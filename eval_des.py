from tensorflow.keras.models import model_from_json, load_model
from cipher.des import DES
from make_data import make_train_data_des
from eval import evaluate

net = load_model("./freshly_trained_nets/DES_3_best_20240112_102015.h5")

des = DES(n_rounds=3)

in_diff = [0x00808200, 0x60000000]

if __name__ == "__main__":
    x, y = make_train_data_des(10000, des, in_diff)
    evaluate(net, x, y)
    # x, y = make_real_differences_data(n_samples, speck5, in_diff)

