using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;

namespace Glenn
{
	class Program
	{
		static void Main(string[] args)
		{
			Console.WriteLine("technically a program");
			var graph = new TFGraph();
			Console.WriteLine("graph! " + graph.ToString());

			//var sessionS = new TFSession
		}
	}
}
