def get_model(model_name, args, loger):
    name = model_name.lower()
    if name == "miles":
        from model.MILES import MILES
        return MILES(args, loger)
    if name == "miles_plus":
        from model.MILES_Plus import MILES_Plus
        return MILES_Plus(args, loger)
    else:
        assert "No such model!"
