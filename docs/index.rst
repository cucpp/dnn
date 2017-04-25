.. CuDNN C++ Wrapper documentation master file, created by
	sphinx-quickstart on Mon Apr 24 11:31:48 2017.
	You can adapt this file completely to your liking, but it should at least
	contain the root `toctree` directive.

.. Welcome to CuDNN C++ Wrapper's documentation!
.. =============================================

API Reference
-------------

.. doxygenclass:: CuDNN::Handle
   :project: cucpp-cudnn
   :members:

.. doxygenclass:: CuDNN::Tensor
   :project: cucpp-cudnn
   :members:

.. doxygenclass:: CuDNN::TensorDescriptor
   :project: cucpp-cudnn
   :members:

.. doxygenclass:: CuDNN::Convolution
   :project: cucpp-cudnn
   :members:

.. doxygenclass:: CuDNN::ConvolutionDescriptor
   :project: cucpp-cudnn
   :members:

.. doxygenclass:: CuDNN::Filter
   :project: cucpp-cudnn
   :members:

.. doxygenclass:: CuDNN::FilterDescriptor
   :project: cucpp-cudnn
   :members:

Exceptions and Error Handling
-----------------------------
.. doxygenclass:: CuDNN::Exception
   :project: cucpp-cudnn
   :members:

.. doxygenfunction:: CuDNN::checkStatus
   :project: cucpp-cudnn

Internal API & Implementation Details
-------------------------------------

.. doxygenclass:: CuDNN::detail::RAII
   :project: cucpp-cudnn
   :members:

.. doxygenclass:: CuDNN::detail::Buffer
   :project: cucpp-cudnn
   :members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
