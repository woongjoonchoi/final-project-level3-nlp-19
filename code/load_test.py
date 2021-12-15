from predict import load_model, get_prediction


def main():

    # 모델 불러오기
    model, tokenizer = load_model()
    # 데이터 받기
    sentence = '유령은 어느 행성에서 지구로 왔는가?'

    prediction = get_prediction(model, tokenizer, sentence)
    print('prediction 출력')
    print(prediction)


main()