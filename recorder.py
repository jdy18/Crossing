import pickle

class Recoder:
    def __init__(self):
        self.save_folder='/data1/gqma/Cell_experiment/'
        self.spikes={}
        self.v = {}
        self.test_accuracy=[]
        self.task_accuracy={}
        self.record_flag=0 #0表示没有记录数据，析构时不重复保存
    def record_spike_v(self,step,task: int,test_accuracy,task_accuracy,z,v):
        try:
            self.spikes['Step'+str(step)]['Task'+str(task)]=z
        except:
            self.spikes['Step' + str(step)]={}
            self.spikes['Step' + str(step)]['Task' + str(task)] = z

        try:
            self.v['Step'+str(step)]['Task'+str(task)]=v
        except:
            self.v['Step' + str(step)]={}
            self.v['Step' + str(step)]['Task' + str(task)] = v


        #meanwhile record test_accuracy and task_accuracy
        if task==1:
            self.test_accuracy.append(test_accuracy)
        try:
            self.task_accuracy['Task'+str(task)].append(task_accuracy)
        except:
            self.task_accuracy['Task' + str(task)] = []
            self.task_accuracy['Task' + str(task)] .append(task_accuracy)

        if self.record_flag==0:
            self.record_flag = 1
    def record_model(self,step,model):
        pickle_file = open(self.save_folder+'model/step_' + str(step)+'_.pkl','wb')
        pickle.dump(model, pickle_file)
        pickle_file.close()
    def load_spike_v(self):
        pickle_file = open(self.save_folder+'data/spikes_.pkl', 'rb')
        self.spikes=pickle.load( pickle_file)
        pickle_file.close()

        pickle_file = open(self.save_folder+'data/v_.pkl', 'rb')
        self.v = pickle.load(pickle_file)
        pickle_file.close()

        pickle_file = open(self.save_folder+'data/test_accuracy_.pkl', 'rb')
        self.test_accuracy=pickle.load( pickle_file)
        pickle_file.close()

        pickle_file = open(self.save_folder+'data/task_accuracy_.pkl', 'rb')
        self.task_accuracy=pickle.load( pickle_file)
        pickle_file.close()


    def load_model(self,step):
        pickle_open = open(self.save_folder+'model/step_' + str(step)+'_.pkl', 'rb')
        model = pickle.load(pickle_open)
        pickle_open.close()
        return model
    def __del__(self):

        if self.record_flag==1:
            pickle_file = open(self.save_folder+'data/spikes_.pkl', 'wb')
            pickle.dump(self.spikes, pickle_file)
            pickle_file.close()

            pickle_file = open(self.save_folder+'data/v_.pkl', 'wb')
            pickle.dump(self.v, pickle_file)
            pickle_file.close()

            pickle_file = open(self.save_folder+'data/test_accuracy_.pkl', 'wb')
            pickle.dump(self.test_accuracy, pickle_file)
            pickle_file.close()

            pickle_file = open(self.save_folder+'data/task_accuracy_.pkl', 'wb')
            pickle.dump(self.task_accuracy, pickle_file)
            pickle_file.close()


