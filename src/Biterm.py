# -*- coding: utf-8 -*-


class Biterm:

    def __init__(self, b_id, w1=None, w2=None, s=None):
        self.b_id = b_id
        # self.wi = 0
        # self.wj = 0
        # self.z = 0

        if w1 is not None and w2 is not None:
            self.wi = min(w1, w2)
            self.wj = max(w1, w2)
        elif w1 is None and w2 is None and s is not None:
            w = s.split(' ')
            self.wi = w[0]
            self.wj = w[1]
            self.z = w[2]

    def equalTo(self, bi):
        return self.b_id == bi.b_id

    # def get_wi(self):
    #     return self.wi
    #
    # def get_wj(self):
    #     return self.wj
    #
    # def get_z(self):
    #     return self.z

    def set_z(self, k):
        self.z = k

    def reset_z(self):
        self.z = -1

    def str(self):
        _str = ""
        _str += str(self.wi) + '\t' + str(self.wj) + '\t\t' + str(self.z)
        return _str
