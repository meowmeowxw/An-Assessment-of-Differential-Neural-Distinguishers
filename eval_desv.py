from tensorflow.keras.models import model_from_json, load_model
from cipher.des_vectorised import DESV
from make_data import make_train_data, make_real_differences_data
from eval import evaluate
import matplotlib.pyplot as plt


net_3_0 = load_model("./freshly_trained_nets/DESV_3_rounds_4008000004000000.h5")
net_3_d0 = [0x40080000, 0x04000000]

net_3_1 = load_model("./freshly_trained_nets/DESV_3_rounds_0080820060000000.h5")
net_3_d1 = [0x00808200, 0x60000000]

net_3_2 = load_model("./freshly_trained_nets/DESV_3_rounds_1960000004000000.h5")
net_3_d2 = [0x19600000, 0x00000000]

des = DESV(n_rounds=3)

if __name__ == "__main__":
    print(f"\n\nEvaluation with ∆ = {list(map(hex, net_3_d0))}")
    x, y = make_train_data(100000, des, net_3_d0)
    acc_0 = evaluate(net_3_0, x, y)
    print(f"\n\nEvaluation of real difference data with ∆ = {list(map(hex, net_3_d0))}")
    x, y = make_real_differences_data(100000, des, net_3_d0)
    evaluate(net_3_0, x, y)

    print(f"\n\nEvaluation with ∆ = {list(map(hex, net_3_d1))}")
    x, y = make_train_data(100000, des, net_3_d1)
    acc_1 = evaluate(net_3_1, x, y)
    print(f"\n\nEvaluation of real difference data with ∆ = {list(map(hex, net_3_d1))}")
    x, y = make_real_differences_data(100000, des, net_3_d1)
    evaluate(net_3_1, x, y)

    print(f"\n\nEvaluation with ∆ = {list(map(hex, net_3_d2))}")
    x, y = make_train_data(100000, des, net_3_d2)
    acc_2 = evaluate(net_3_2, x, y)
    print(f"\n\nEvaluation of real difference data with ∆ = {list(map(hex, net_3_d2))}")
    x, y = make_real_differences_data(100000, des, net_3_d2)
    evaluate(net_3_2, x, y)


    labels = [r'$\Delta_0$', r'$\Delta_1$', r'$\Delta_2$']
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    plt.bar(labels, [acc_0, acc_1, acc_2], color=colors)

# Set y-axis limits
    plt.ylim(0.5, 1)

# Set axis labels and title
    plt.xlabel('Deltas')
    plt.ylabel('Accuracy')
    plt.title('Comparison')
    plt.show()
