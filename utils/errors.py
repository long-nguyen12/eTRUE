

def optional_result(errors, label, operation):
    try:
        return operation()
    except Exception as error:
        errors.append(label + ": " + str(error))
        return None
