import os
import py_compile

def check_syntax(directory):
    skip_dirs = {'.venv', 'venv', '__pycache__'}
    errors = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                try:
                    py_compile.compile(path, doraise=True)
                except py_compile.PyCompileError as e:
                    errors.append(str(e))
    return errors

if __name__ == "__main__":
    errs = check_syntax(r'e:\DocumentVerify\ai-ml-service')
    if errs:
        print("\n".join(errs))
    else:
        print("No syntax errors found.")
