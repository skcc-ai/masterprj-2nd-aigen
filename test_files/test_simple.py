def simple_function():
    """간단한 테스트 함수"""
    return "Hello, World!"

class SimpleClass:
    """간단한 테스트 클래스"""
    
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value

def main():
    obj = SimpleClass()
    result = simple_function()
    return obj.get_value(), result

if __name__ == "__main__":
    result = main()
    print(f"결과: {result}")
