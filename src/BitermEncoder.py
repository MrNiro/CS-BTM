from transformers import BertModel, BertTokenizer
import torch

device = torch.device('cuda')


class BitermBert:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print("Initializing Biterm Encoder...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)  # .to(device)
        self.bert_model = BertModel.from_pretrained(model_name).to(device)
        print("Biterm Encoder Init Done!")

    def encode(self, text):
        token = self.tokenizer.encode(text)
        result = self.bert_model(torch.tensor(token).unsqueeze(0).to(device))
        return result[1]    # size(1 * 768)

    def cal_biterm_similarity(self, b1: tuple, b2: tuple):
        w1, w2 = b1[0], b1[1]
        w3, w4 = b2[0], b2[1]

        v1 = self.encode(w1)
        v2 = self.encode(w2)
        v3 = self.encode(w3)
        v4 = self.encode(w4)

        similarity = (torch.cosine_similarity(v1, v3) +
                      torch.cosine_similarity(v1, v4) +
                      torch.cosine_similarity(v2, v3) +
                      torch.cosine_similarity(v2, v4)) / 4
        return float(similarity)

    @staticmethod
    def cal_vector_similarity(b1, b2):
        v1, v2 = b1[0], b1[1]
        v3, v4 = b2[0], b2[1]

        similarity = (torch.cosine_similarity(v1, v3) +
                      torch.cosine_similarity(v1, v4) +
                      torch.cosine_similarity(v2, v3) +
                      torch.cosine_similarity(v2, v4)) / 4
        return float(similarity)


def simple_test():
    # test_model_name = 'bert-base-multilingual-cased'
    test_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    # test_model_name = 'bert-base-chinese'
    # test_model_name = 'bert-base-uncased'

    # model_path = './model/multi_cased_L-12_H-768_A-12/'

    tokenizer = BertTokenizer.from_pretrained(test_model_name)
    bert_model = BertModel.from_pretrained(test_model_name)      # 这地方好坑,用model path的话模型与源码不一定兼容

    # t1 = tokenizer.encode('vaccines')
    # t2 = tokenizer.encode("vaccine")
    # t3 = tokenizer.encode('vaccination')
    # t4 = tokenizer.encode('mainland')

    t1 = tokenizer.encode('Nancy Pelosi’s visit to Taiwan is expected to cost tax payers over $90 million for security, allocation of US military presence, and more.')
    t2 = tokenizer.encode("Nancy Pelosi is willing to risk starting a war with China so that she can make massive profits on her husband's insider trading deals on computer chips.#Pelosi #Taiwan.")
    t3 = tokenizer.encode('What a great tournament! Incredible advert for womens football and sport in general. Inspirational stuff. Congrats @Lionesses')
    t4 = tokenizer.encode('Football is a simple game. 22 women chase a ball for 90 minutes and, at the end, England actually win. Congratulations @lionesses. Fabulous.')

    input_1 = torch.tensor(t1).unsqueeze(0)
    input_2 = torch.tensor(t2).unsqueeze(0)
    input_3 = torch.tensor(t3).unsqueeze(0)
    input_4 = torch.tensor(t4).unsqueeze(0)

    r1 = bert_model(input_1)
    r2 = bert_model(input_2)
    r3 = bert_model(input_3)
    r4 = bert_model(input_4)

    print(r1[1].size())

    dis_1_2 = torch.cosine_similarity(r1[1], r2[1])
    dis_1_3 = torch.cosine_similarity(r1[1], r3[1])
    dis_3_4 = torch.cosine_similarity(r3[1], r4[1])

    print(dis_1_2)
    print(dis_1_3)
    print(dis_3_4)


if __name__ == '__main__':
    simple_test()
