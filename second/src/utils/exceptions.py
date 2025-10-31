class BasicModelException(Exception):
    """自定义异常基类"""
    pass


class InvalidBackboneError(BasicModelException):
    """无效的骨干网络错误"""
    pass


class InvalidDatasetSelection(BasicModelException):
    """无效的数据集选择错误"""
    pass

class InvalidDatasetClassesNum(BasicModelException):
    """无效的数据集选择错误"""
    pass
