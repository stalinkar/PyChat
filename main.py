from Model import script, words, classes, data

# Press the green button in the gutter to run the script.
# from Model.script import words, classes

if __name__ == '__main__':
    print("Press 0 if you don't want to chat with our ChatBot.")
    result = ""
    while True:
        message = input("")
        if message == "0":
            break
        else:
            if result == "Really!!! Please correct me":
                result = script.update_data(data, message)
                print(result)
            else:
                intents = script.pred_class(message, words, classes)
                result = script.get_response(intents, data)
                print(result)
