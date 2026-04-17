def get_model(model_name, args):
    name = model_name.lower()
    if name == "dakt":
        from models.DAKT import DAKT
        return DAKT(args)
    else:
        assert 0
