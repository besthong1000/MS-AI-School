import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


from torch.utils.data import DataLoader
from torchvision.models import resnet18
from ex_02_customdata import CustomDataset_ex02
from tqdm import tqdm

def main() :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model setting
    model = resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)

# .pt load
    model.load_state_dict(torch.load(f="./art_best.pt"))
# print(list(model.parameters())) 추천x 어쩌피 봐도 모른다.


# 랜덤한 값은 제외. 변하면 안된다. 학습할 때 한 것들.
# 학습할 때 기록할 것 : 하이퍼파라미터, 어그멘테이션 뭐썼는지.러닝 레이트, 옵티마이저
    val_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    test_dataset = CustomDataset_ex02("./data_art/val/",
                                      val_transforms)
    # for i in test_dataset :
    #     print(i)
    # 잘 넘어오는지 확인하자!


    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # for image, label in test_loader:
    #     print(image,label)

    model.to(device)
    model.eval()

    correct = 0
# from tqdm import tqdm
    import cv2
    label_dict = {0: "Abstract", 1: "Cubist", 2 : "Expressionist",
                  3: "Impressionist", 4: "Landscape", 5: "Pop Art",
                  6: "Portrait", 7: "Realist", 8: "Still Life",
                  9: "Surrealist"}

    with torch.no_grad() :
        for data, target , path in tqdm(test_loader) :
            # tqdm으로 진행상황을 알수 있음, tqdm과 ()는 생략가능

            target_=target.item()
            data,  target = data.to(device), target.to(device)  # cuda 로 학습되어 있음
            output = model(data)
            # 숫자가 큰 걸로 예측한다.

            pred = output.argmax(dim=1, keepdim=True)
            # print("pred>>" ,pred.item(), path) # 예측값, 어떻게 예측을 했는지 알수 있다.


            target_label = label_dict[target_]
            true_label_text = f"true : {target_label}"


            img = cv2.imread(path[0])
            img = cv2.resize(img, (500,500))
            pred_label = label_dict[pred.item()]
            pred_text = f"pred : {pred_label}"
            img = cv2.rectangle(img, (0,0), (500,100), (255,255,255), -1)
            img = cv2.putText(img, pred_text, (0,30),
                              cv2.FONT_ITALIC, 1, (255,0,0), 2 )
            img = cv2.putText(img, true_label_text, (0, 75),
                              cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

            cv2.imshow("test", img)
            if cv2.waitKey() == ord('q') :
                exit()

            correct += pred.eq(target.view_as(pred)).sum().item()

    print("test set : Acc {}/{} [{:.0f}]%\n".format(
        correct, len(test_loader.dataset),
        100*correct / len(test_loader.dataset)
    ))


if __name__ == "__main__" :
    main()