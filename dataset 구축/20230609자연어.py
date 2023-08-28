#pip install transformers

from transformers import pipeline

#pipline() 함수를 호출하면서 관심 작업 이름을 전달해 파이프라인 객체 생성
classifiter = pipeline("text-classification")
 


text = """Dear Amazon, last week I ordered an Optimus Prime action figure \from your online store in Germany. Unfortunately, when I opened the package, \I discovered to my horror that I had been sent an action figure of Megatron \instead! As a lifelong enemy of the Decepticons, I hope you can understand my \dilemma. To resolve the issue, I demand an exchange of Megatron for the \Optimus Prime figure I ordered. Enclosed are copies of my records concerning \this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""

outputs = classifiter(text)
for output in outputs :
    print(output["label"], output["score"])