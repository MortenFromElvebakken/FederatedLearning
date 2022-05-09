
class logger():
    def __init__(self, args):
        if args.logger == 'neptune':
            import neptune.new as neptune
            run = neptune.init(project='Federated_learning/fedAVG',
                               api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ODA1YThjNC1hNWViLTQxZDEtOWNjOS0zYmY5ZTA1NDQ1NTEifQ==')
            # use run[...].log(...) to log to neptune
            run['parameters'] = args
        else:
            # implement csv logger?
            run = None
    #def logmetric(self):


    #def logparams(self):
