from tensorflow.keras.models import model_from_json, load_model
from cipher.des_vectorised import DESV
from make_data import make_train_data, make_real_differences_data
from eval import evaluate


net_3_0 = load_model("./freshly_trained_nets/DESV_3_rounds_4008000004000000.h5")
net_3_d0 = [0x40080000, 0x04000000]

net_3_2 = load_model("./freshly_trained_nets/DESV_3_rounds_1960000004000000.h5")
net_3_d2 = [0x19600000, 0x00000000]

des = DESV(n_rounds=3)

if __name__ == "__main__":
    print(f"Evaluation with ∆ = {list(map(hex, net_3_d0))}")
    x, y = make_train_data(100000, des, net_3_d0)
    evaluate(net_3_0, x, y)
    print(f"\n\nEvaluation of real difference data with ∆ = {list(map(hex, net_3_d0))}")
    x, y = make_real_differences_data(100000, des, net_3_d0)
    evaluate(net_3_0, x, y)

    print(f"\n\nEvaluation with ∆ = {list(map(hex, net_3_d2))}")
    x, y = make_train_data(100000, des, net_3_d2)
    evaluate(net_3_2, x, y)
    print(f"\n\nEvaluation of real difference data with ∆ = {list(map(hex, net_3_d2))}")
    x, y = make_real_differences_data(100000, des, net_3_d2)
    evaluate(net_3_2, x, y)


