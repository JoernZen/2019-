#FROM python:alpine
#from conda/miniconda3-centos7
#from conda/miniconda3-centos6
FROM centos:centos7
MAINTAINER Conda Development Team <conda@continuum.io>

#RUN yum -y update \
# && yum install epel-release -y \
# && yum install https://centos7.iuscommunity.org/ius-release.rpm -y \
# && yum install python36u -y \
# && yum install python36u-devel -y \
# && yum install python36u-pip -y \
# && echo 'alias python='/usr/bin/python3.6''>>~/.bashrc \
# && yum clean all 
RUN yum -y update \
    && yum -y install curl bzip2 \
    && curl -sSL https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local/ \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3.6 \
    && conda clean --all --yes \
    && rpm -e --nodeps curl bzip2 \
    && yum clean all
workdir /code
copy . /code
run pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt --timeout=1000
#run pip install pandas --timeout=1000
