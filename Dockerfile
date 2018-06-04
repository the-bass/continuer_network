FROM continuumio/anaconda3:5.1.0

RUN conda install pytorch-cpu torchvision-cpu -c pytorch

RUN pip install torch_testing

WORKDIR /app

ADD dataset_parties_dev dataset_parties_dev
RUN pip install -e ./dataset_parties_dev

ADD torch_state_control_dev torch_state_control_dev
RUN pip install -e ./torch_state_control_dev

ADD coach_dev coach_dev
RUN pip install -e ./coach_dev

ADD sequences_creator_dev sequences_creator_dev
RUN pip install -e ./sequences_creator_dev
