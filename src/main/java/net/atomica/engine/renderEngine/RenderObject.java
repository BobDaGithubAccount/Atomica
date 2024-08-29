package net.atomica.engine.renderEngine;

import net.atomica.engine.entities.Entity;
import net.atomica.engine.models.TexturedModel;

import java.util.HashMap;
import java.util.Map.Entry;

public class RenderObject {

	public HashMap<String, Entity> instances;
	public TexturedModel model;
	
	public RenderObject(HashMap<String, Entity> instances, TexturedModel model) {
		if(instances == null) {
			this.instances = new HashMap<String, Entity>();
		}
		else {
			this.instances = instances;
		}
		this.model = model;
	}
	
	@Override
	public String toString() {
		String toReturn = "{model=" + model.toString() + ";instances={";
		for(Entry<String, Entity> e : instances.entrySet()) {
			toReturn += e.getKey() + "=" + e.getValue().toString() + ";";
		}
		StringBuffer sb = new StringBuffer(toReturn);   
		sb.deleteCharAt(sb.length()-1);
		toReturn = sb.toString();
		toReturn += "}}";
		return toReturn;
	}
	
}
