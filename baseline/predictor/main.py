from baseline.predictor.predictor import Predictor
from judger import judger

if __name__ == '__main__':
    str='温州市鹿城区人民检察院指控，2013年2月份以来，被告人郑某在担任温州华侨永利会KTV公关经理期间，先后介绍该KTV小姐郭某、赵某在华侨饭店客房内向袁某、胡某等人提供卖淫服务，其中介绍郭某卖淫2次、介绍赵某卖淫3次。'
    pre=Predictor()
    ans=pre.predict(content=str)

